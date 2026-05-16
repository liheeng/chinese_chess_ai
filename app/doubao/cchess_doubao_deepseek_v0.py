# 优化版：动态学习率 + 快速收敛
# - 代码框架由doubao提供
# - 参数动态调节由deepseek v4 flash提供
# - 不限制轮次训练，随时停，随时继续

# 棋谱：99771盘，样本数811万
# Full 26 channels to train:
# STEP_CHANNELS * HISTORY_LEN + PIECE_PLANES + SIDE_CHANNELS
# Parameters:
# batch size: 256
# CNN: 4 层
# Transformer：nhead=4（增强全局棋力理解）

import numpy as np
import torch
import torch.nn as nn
import re
import os
import signal
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

torch.mps.empty_cache()  # 清空MPS缓存

# 文件路径
file_name = os.path.splitext(os.path.basename(__file__))[0]
BASE_DIR = "./data"
CACHE_PATH = f"{BASE_DIR}/xiangqi_dataset_cache_{file_name}.pt"
CHECKPOINT_PATH = f"{BASE_DIR}/xiangqi_ckpt_latest_{file_name}.pth"
BEST_MODEL_PATH = f"{BASE_DIR}/xiangqi_best_{file_name}.pth"

# ===================== 全局配置【完整26通道】 =====================
HISTORY_LEN = 4
STEP_CHANNELS = 2
PIECE_PLANES = 14    # 14种棋子独立平面
SIDE_CHANNELS = 4    # 红回合、黑回合、九宫、河界
# 总输入通道：历史轨迹(8) + 棋子盘面(14) + 规则平面(4) = 26
IN_CHANNELS = STEP_CHANNELS * HISTORY_LEN + PIECE_PLANES + SIDE_CHANNELS
COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}

# 棋子定义
EMPTY = 0
RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P = 1, 2, 3, 4, 5, 6, 7
BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P = 8, 9, 10, 11, 12, 13, 14

# 红/黑棋子分组
RED_PIECES = {RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P}
BLK_PIECES = {BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P}

# 所有棋子编号列表，用于生成14层平面
ALL_PIECE_IDS = [
    RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P,
    BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P,
]

FEN_PIECE = {
    "K": RED_K,
    "R": RED_R,
    "N": RED_N,
    "C": RED_C,
    "E": RED_E,
    "A": RED_A,
    "P": RED_P,
    "k": BLK_K,
    "r": BLK_R,
    "n": BLK_N,
    "c": BLK_C,
    "e": BLK_E,
    "a": BLK_A,
    "p": BLK_P,
}


# ===================== 棋盘点面编码（14层） =====================
def board_to_piece_planes(board_arr):
    """将 (9,10) 棋盘编码为 (14,9,10) 二值平面，每类棋子单独一层"""
    planes = np.zeros((PIECE_PLANES, 9, 10), dtype=np.uint8)
    for idx, pid in enumerate(ALL_PIECE_IDS):
        planes[idx] = (board_arr == pid).astype(np.uint8)
    return planes


# ===================== 规则平面（4层） =====================
def get_side_planes(turn: int):
    """
    生成规则平面：红方回合 + 黑方回合 + 九宫掩码 + 河界掩码
    turn: 0=红方, 1=黑方
    return: (4, 9, 10) uint8
    """
    red_turn = (
        np.ones((1, 9, 10), dtype=np.uint8)
        if turn == 0
        else np.zeros((1, 9, 10), dtype=np.uint8)
    )
    blk_turn = (
        np.ones((1, 9, 10), dtype=np.uint8)
        if turn == 1
        else np.zeros((1, 9, 10), dtype=np.uint8)
    )

    # 九宫禁区掩码
    palace = np.zeros((1, 9, 10), dtype=np.uint8)
    palace[0, 3:6, 0:3] = 1   # 红方九宫
    palace[0, 3:6, 7:10] = 1  # 黑方九宫

    # 河界分区：红方区域=1，黑方区域=0
    river = np.zeros((1, 9, 10), dtype=np.uint8)
    river[0, :, :5] = 1  # 红方半场 (row 0-4)

    return np.concatenate([red_turn, blk_turn, palace, river], axis=0)


# ===================== 合法走法掩码（from/to 双掩码） =====================
def in_border(c, r):
    return 0 <= c < 9 and 0 <= r < 10


def is_red(pid):
    return 1 <= pid <= 7


def is_black(pid):
    return 8 <= pid <= 14


def get_pseudo_legal_moves(board, c0, r0):
    """伪合法走法（不检测将军），与 xiangqi_base.py 规则一致"""
    pid = board[c0, r0]
    if pid == EMPTY:
        return []
    moves = []

    # 将/帅
    if pid in (RED_K, BLK_K):
        for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if is_red(pid) and not (3 <= c1 <= 5 and 0 <= r1 <= 2):
                continue
            if is_black(pid) and not (3 <= c1 <= 5 and 7 <= r1 <= 9):
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    # 士
    elif pid in (RED_A, BLK_A):
        for dc, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if is_red(pid) and not (3 <= c1 <= 5 and 0 <= r1 <= 2):
                continue
            if is_black(pid) and not (3 <= c1 <= 5 and 7 <= r1 <= 9):
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    # 象
    elif pid in (RED_E, BLK_E):
        jumps = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        eyes = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for idx, (dc, dr) in enumerate(jumps):
            ec, er = c0 + eyes[idx][0], r0 + eyes[idx][1]
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1) or not in_border(ec, er):
                continue
            if board[ec, er] != EMPTY:
                continue
            if is_red(pid) and r1 >= 5:
                continue
            if is_black(pid) and r1 <= 4:
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    # 马
    elif pid in (RED_N, BLK_N):
        jumps = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                 (1, 2), (1, -2), (-1, 2), (-1, -2)]
        blocks = [(1, 0), (1, 0), (-1, 0), (-1, 0),
                  (0, 1), (0, -1), (0, 1), (0, -1)]
        for idx, (dc, dr) in enumerate(jumps):
            bc, br = c0 + blocks[idx][0], r0 + blocks[idx][1]
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if board[bc, br] != EMPTY:
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    # 车
    elif pid in (RED_R, BLK_R):
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1, r1 = c0 + dc, r0 + dr
            while in_border(c1, r1):
                if board[c1, r1] != EMPTY:
                    if (is_red(pid) and board[c1, r1] not in RED_PIECES) or \
                       (is_black(pid) and board[c1, r1] not in BLK_PIECES):
                        moves.append((c1, r1))
                    break
                moves.append((c1, r1))
                c1 += dc
                r1 += dr

    # 炮
    elif pid in (RED_C, BLK_C):
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1, r1 = c0 + dc, r0 + dr
            jumped = False
            while in_border(c1, r1):
                if not jumped:
                    if board[c1, r1] == EMPTY:
                        moves.append((c1, r1))
                    else:
                        jumped = True
                else:
                    if board[c1, r1] != EMPTY:
                        if (is_red(pid) and board[c1, r1] not in RED_PIECES) or \
                           (is_black(pid) and board[c1, r1] not in BLK_PIECES):
                            moves.append((c1, r1))
                        break
                c1 += dc
                r1 += dr

    # 兵/卒
    elif pid in (RED_P, BLK_P):
        if is_red(pid):
            dirs = [(0, -1)]
            if r0 <= 4:
                dirs += [(1, 0), (-1, 0)]
        else:
            dirs = [(0, 1)]
            if r0 >= 5:
                dirs += [(1, 0), (-1, 0)]
        for dc, dr in dirs:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    return moves


def get_legal_mask(board, turn):
    """生成完整 legal from/to 掩码 (90,)：合法=0.0，非法=-inf"""
    from_mask = np.full(90, -np.inf, dtype=np.float32)
    to_mask = np.full(90, -np.inf, dtype=np.float32)
    allow_pieces = RED_PIECES if turn == 0 else BLK_PIECES

    for c in range(9):
        for r in range(10):
            pid = board[c, r]
            if pid not in allow_pieces:
                continue
            fid = c * 10 + r
            from_mask[fid] = 0.0
            for c1, r1 in get_pseudo_legal_moves(board, c, r):
                tid = c1 * 10 + r1
                to_mask[tid] = 0.0
    return from_mask, to_mask


# ===================== 象棋棋盘 =====================
class XiangqiBoard:
    def __init__(self):
        self.board = np.zeros((9, 10), dtype=int)
        self.history = []
        self.turn = 0  # 🔥 新增：回合记录 0=红 1=黑

    def load_fen(self, fen):
        self.board.fill(EMPTY)
        self.turn = 0 if fen.split()[1] == "w" else 1
        fen_board = fen.split()[0]
        rows = fen_board.split("/")
        for r_idx, row in enumerate(rows):
            c_idx = 0
            for ch in row:
                if ch.isdigit():
                    c_idx += int(ch)
                else:
                    if ch in FEN_PIECE and c_idx < 9:
                        self.board[c_idx, r_idx] = FEN_PIECE[ch]
                c_idx += 1

    def push(self, move):
        fc, fr, tc, tr = move
        self.history.append((fc, fr, tc, tr))
        self.board[tc, tr] = self.board[fc, fr]
        self.board[fc, fr] = EMPTY
        self.turn = 1 - self.turn  # 切换回合


# ===================== 棋谱解析 =====================
def iccs_move_to_pos(move_str):
    """将 ICCS 走法字符串转为坐标 (fc,fr,tc,tr)，格式错误返回 None"""
    if "-" not in move_str:
        return None
    parts = move_str.strip().split("-")
    if len(parts) != 2:
        return None
    frm, to = parts
    if len(frm) < 2 or len(to) < 2:
        return None
    try:
        fc = COL_MAP[frm[0].upper()]
        fr = int(frm[1])
        tc = COL_MAP[to[0].upper()]
        tr = int(to[1])
    except (KeyError, ValueError, IndexError):
        return None
    if not (0 <= fc < 9 and 0 <= tc < 9 and 0 <= fr < 10 and 0 <= tr < 10):
        return None
    return (fc, fr, tc, tr)


def parse_iccs_pgn(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    game_blocks = re.split(r"\n\n(?=\[Game)", content)
    games = []
    for block in game_blocks:
        if "[FEN" not in block or "1." not in block:
            continue
        fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', block)
        if not fen_match:
            continue
        fen = fen_match.group(1)
        move_part = re.sub(r"\[.*?\]", "", block)
        move_part = re.sub(r"\d+\.", "", move_part)
        move_part = re.sub(r"\s+", " ", move_part).strip()
        move_tokens = [m for m in move_part.split() if "-" in m]
        games.append({"fen": fen, "moves": move_tokens})
    return games


# ===================== 数据编码【完整26通道 + 合法掩码】 =====================
EMPTY_STEP = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)


def encode_move(fc, fr, tc, tr):
    step = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)
    step[0, fc, fr] = 1
    step[1, tc, tr] = 1
    return step


def iccs_games_to_dataset(games, max_games):
    X, y_from, y_to, mask_from, mask_to = [], [], [], [], []
    target_shape = (IN_CHANNELS, 9, 10)

    for game in tqdm(games[:max_games], desc="生成训练数据"):
        board = XiangqiBoard()
        try:
            board.load_fen(game["fen"])
        except Exception:
            continue

        for mv_str in game["moves"]:
            try:
                move = iccs_move_to_pos(mv_str)
                if move is None:
                    continue
                fc, fr, tc, tr = move

                # 历史轨迹 (0..7通道)
                hist = board.history
                pad = [EMPTY_STEP] * max(0, HISTORY_LEN - len(hist))
                real_steps = [encode_move(*m) for m in hist[-HISTORY_LEN:]]
                state_trace = np.concatenate(pad + real_steps, axis=0)

                # 棋子平面 (8..21通道)
                state_pieces = board_to_piece_planes(board.board)

                # 规则平面 (22..25通道)
                state_side = get_side_planes(board.turn)

                # 拼接完整状态
                state = np.concatenate([state_trace, state_pieces, state_side], axis=0)

                if state.shape != target_shape:
                    continue

                # 合法走法掩码
                m_f, m_t = get_legal_mask(board.board, board.turn)

                fid = fc * 10 + fr
                tid = tc * 10 + tr

                X.append(state)
                y_from.append(fid)
                y_to.append(tid)
                mask_from.append(m_f)
                mask_to.append(m_t)

                board.push(move)

            except Exception:
                continue

    return (
        np.array(X, dtype=np.uint8),            # 二值平面用 uint8 (1B) → 节省75%
        np.array(y_from, dtype=np.uint8),        # 0~89 用 uint8 (1B)
        np.array(y_to, dtype=np.uint8),          # 0~89 用 uint8 (1B)
        np.array(mask_from, dtype=np.float16),   # 0/-inf 用 float16 (2B)
        np.array(mask_to, dtype=np.float16),     # 0/-inf 用 float16 (2B)
    )


# ===================== 缓存工具 =====================
def save_cache(data, path):
    torch.save(data, path)


def load_cache(path):
    return torch.load(path)


# ===================== 模型【仅修改输入通道】 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=90):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class HybridXiangqiModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 🔥 自动适配新输入通道
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.from_head = nn.Linear(d_model, 90)
        self.to_head = nn.Linear(d_model, 90)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.from_head(x).mean(1), self.to_head(x).mean(1)


# ===================== 断点保存/加载 =====================
def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,  # 🔥 新增：保存最优loss
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    # 🔥 新增：加载best_loss，兼容旧断点
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"\n✅ 加载断点成功，从第 {start_epoch} 轮继续训练 | 历史最优Loss: {best_loss:.3f}")
    return start_epoch, best_loss


# ===================== 优雅退出 =====================
stop_flag = False


def handle_exit(signum, frame):
    global stop_flag
    print("\n\n🛑 停止信号收到，本轮结束后保存退出...")
    stop_flag = True


signal.signal(signal.SIGINT, handle_exit)

# ===================== 主程序 =====================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 训练设备: {device}")
    print("🚀 优化版：动态学习率 + Side Plane + Piece Plane + History + Legal Move")

    PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"

    # 旧缓存不兼容新26通道，自动重新生成
    if os.path.exists(CACHE_PATH):
        try:
            data = load_cache(CACHE_PATH)
            if len(data) == 5:
                X, yf, yt, mf, mt = data
            else:
                raise ValueError("旧缓存格式")
            assert X.shape[1] == IN_CHANNELS, f"通道数不匹配: {X.shape[1]} != {IN_CHANNELS}"
            print(f"✅ 发现兼容缓存，快速加载... 样本数: {len(X)}")
        except Exception as e:
            print(f"⚠️  缓存不兼容 ({e})，重新生成26通道数据集...")
            if os.path.exists(CACHE_PATH):
                os.remove(CACHE_PATH)
            games = parse_iccs_pgn(PGN_PATH)
            X, yf, yt, mf, mt = iccs_games_to_dataset(games, max_games=len(games))
            X = torch.tensor(X)
            yf = torch.tensor(yf, dtype=torch.long)
            yt = torch.tensor(yt, dtype=torch.long)
            mf = torch.tensor(mf, dtype=torch.float32)
            mt = torch.tensor(mt, dtype=torch.float32)
            save_cache((X, yf, yt, mf, mt), CACHE_PATH)
            print(f"✅ 数据集重建完成，样本数: {len(X)}")
    else:
        print("❌ 首次运行，生成训练数据...")
        games = parse_iccs_pgn(PGN_PATH)
        X, yf, yt, mf, mt = iccs_games_to_dataset(games, max_games=len(games))
        X = torch.tensor(X)
        yf = torch.tensor(yf, dtype=torch.long)
        yt = torch.tensor(yt, dtype=torch.long)
        mf = torch.tensor(mf, dtype=torch.float32)
        mt = torch.tensor(mt, dtype=torch.float32)
        save_cache((X, yf, yt, mf, mt), CACHE_PATH)
        print(f"✅ 数据集缓存完成，样本数: {len(X)}")

    train_loader = DataLoader(
        TensorDataset(X, yf, yt, mf, mt),
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # 模型初始化
    model = HybridXiangqiModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss().to(device)
    start_epoch = 0
    best_loss = float("inf")

    # 加载断点
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_loss = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )

    print("\n🚀 无限训练模式启动（26通道 + Legal Move 掩码）\n")
    current_epoch = start_epoch
    running_loss = 0.0
    while True:
        if stop_flag:
            break

        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}")
        for batch in pbar:
            if stop_flag:
                break
            bx, bf, bt, bm_f, bm_t = batch
            bx = bx.to(device, non_blocking=True).float()   # uint8 → float32
            bf = bf.to(device, non_blocking=True).long()    # uint8 → long
            bt = bt.to(device, non_blocking=True).long()    # uint8 → long
            bm_f = bm_f.to(device, non_blocking=True).float()  # float16 → float32
            bm_t = bm_t.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)

            # 🔥 合法走法掩码：非法位置 logit → -inf，softmax 后概率为0
            pred_f = pred_f + bm_f
            pred_t = pred_t + bm_t

            loss = criterion(pred_f, bf) + criterion(pred_t, bt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss = running_loss * 0.98 + loss.item() * 0.02
            total_loss += loss.item()
            steps += 1
            avg_loss = total_loss / steps

            pbar.set_postfix({
                "avg": f"{avg_loss:.3f}",
                "run": f"{running_loss:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {current_epoch} | Loss: {avg_loss:.3f} | LR: {current_lr:.6f}")
        save_checkpoint(model, optimizer, scheduler, current_epoch, best_loss, CHECKPOINT_PATH)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 新最优模型 | Best Loss: {best_loss:.3f}\n")

        current_epoch += 1

    save_checkpoint(model, optimizer, scheduler, current_epoch - 1, best_loss, CHECKPOINT_PATH)
    print("\n🏆 训练已安全停止，下次可直接续训")
