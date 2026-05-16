# 优化版：动态学习率 + 快速收敛
# 棋谱：99771盘，样本数811万
# CNN：4 层（增强特征提取）
# Transformer：nhead=4（增强全局棋力理解）
# 不限制轮次训练，随时停，随时继续
import numpy as np
import torch
import torch.nn as nn
import re
import os
import signal
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

torch.mps.empty_cache()  # 清空MPS缓存

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
    RED_K,
    RED_R,
    RED_N,
    RED_C,
    RED_E,
    RED_A,
    RED_P,
    BLK_K,
    BLK_R,
    BLK_N,
    BLK_C,
    BLK_E,
    BLK_A,
    BLK_P,
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

# ===================== 全局配置【完整版标准通道】 =====================
HISTORY_LEN = 4
STEP_CHANNELS = 2
# 14个棋子独立平面
PIECE_PLANES = 14
# Side 规则平面：红回合、黑回合、九宫、河界
SIDE_CHANNELS = 4

# 总输入通道：历史轨迹 + 棋子盘面 + 规则平面
IN_CHANNELS = STEP_CHANNELS * HISTORY_LEN + PIECE_PLANES + SIDE_CHANNELS


# 文件路径
CACHE_PATH = "xiangqi_dataset_cache_v4-2.pt"
CHECKPOINT_PATH = "ckpt_latest_v4-2.pth"
BEST_MODEL_PATH = "xiangqi_best_v4-2.pth"


# ===================== 缺失：象棋规则 Legal Move 函数 =====================
def in_border(c, r):
    return 0 <= c < 9 and 0 <= r < 10


def in_red_palace(c, r):
    return 3 <= c <= 5 and 0 <= r <= 2


def in_blk_palace(c, r):
    return 3 <= c <= 5 and 7 <= r <= 9


def is_red(pid):
    return 1 <= pid <= 7


def is_black(pid):
    return 8 <= pid <= 14


def get_legal_moves(board, c0, r0):
    pid = board[c0, r0]
    if pid == EMPTY:
        return []
    moves = []

    # 将/帅
    if pid in (RED_K, BLK_K):
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dc, dr in dirs:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if is_red(pid) and not in_red_palace(c1, r1):
                continue
            if is_black(pid) and not in_blk_palace(c1, r1):
                continue
            if is_red(pid) and board[c1, r1] in RED_PIECES:
                continue
            if is_black(pid) and board[c1, r1] in BLK_PIECES:
                continue
            moves.append((c1, r1))

    # 士/仕
    elif pid in (RED_A, BLK_A):
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dc, dr in dirs:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if is_red(pid) and not in_red_palace(c1, r1):
                continue
            if is_black(pid) and not in_blk_palace(c1, r1):
                continue
            if is_red(pid) and board[c1, r1] in RED_PIECES:
                continue
            if is_black(pid) and board[c1, r1] in BLK_PIECES:
                continue
            moves.append((c1, r1))

    # 象/相
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
            if is_red(pid) and board[c1, r1] in RED_PIECES:
                continue
            if is_black(pid) and board[c1, r1] in BLK_PIECES:
                continue
            moves.append((c1, r1))

    # 马
    elif pid in (RED_N, BLK_N):
        jumps = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        block = [(1, 0), (1, 0), (-1, 0), (-1, 0), (0, 1), (0, -1), (0, 1), (0, -1)]
        for idx, (dc, dr) in enumerate(jumps):
            bc, br = c0 + block[idx][0], r0 + block[idx][1]
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if board[bc, br] != EMPTY:
                continue
            if is_red(pid) and board[c1, r1] in RED_PIECES:
                continue
            if is_black(pid) and board[c1, r1] in BLK_PIECES:
                continue
            moves.append((c1, r1))

    # 车
    elif pid in (RED_R, BLK_R):
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1, r1 = c0 + dc, r0 + dr
            while in_border(c1, r1):
                if board[c1, r1] != EMPTY:
                    if (is_red(pid) and board[c1, r1] not in RED_PIECES) or (
                        is_black(pid) and board[c1, r1] not in BLK_PIECES
                    ):
                        moves.append((c1, r1))
                    break
                moves.append((c1, r1))
                c1 += dc
                r1 += dr

    # 炮
    elif pid in (RED_C, BLK_C):
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1, r1 = c0 + dc, r0 + dr
            jump = False
            while in_border(c1, r1):
                if not jump:
                    if board[c1, r1] == EMPTY:
                        moves.append((c1, r1))
                    else:
                        jump = True
                else:
                    if board[c1, r1] != EMPTY:
                        if (is_red(pid) and board[c1, r1] not in RED_PIECES) or (
                            is_black(pid) and board[c1, r1] not in BLK_PIECES
                        ):
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
            if is_red(pid) and board[c1, r1] in RED_PIECES:
                continue
            if is_black(pid) and board[c1, r1] in BLK_PIECES:
                continue
            moves.append((c1, r1))
    return moves


def get_legal_mask(board, turn):
    # 0=非法, 1=合法
    from_mask = np.zeros(90, dtype=np.uint8)
    to_mask = np.zeros(90, dtype=np.uint8)
    allow_pieces = RED_PIECES if turn == 0 else BLK_PIECES

    for c in range(9):
        for r in range(10):
            pid = board[c, r]
            if pid not in allow_pieces:
                continue

            fid = c * 10 + r
            from_mask[fid] = 1

            for c1, r1 in get_legal_moves(board, c, r):
                tid = c1 * 10 + r1  # ✅ 修复：r → r1
                to_mask[tid] = 1

    return from_mask, to_mask


# ===================== 工具：生成14层棋子二值平面 =====================
def board_to_piece_planes(board_arr):
    """
    输入 (9,10) 整数棋盘
    输出 (14,9,10) 二值平面，每一类棋子单独一层
    """
    planes = np.zeros((14, 9, 10), dtype=np.uint8)  # ✅ uint8
    for idx, pid in enumerate(ALL_PIECE_IDS):
        planes[idx] = (board_arr == pid).astype(np.uint8)  # ✅ uint8
    return planes


# ===================== 工具：Side 规则平面 =====================
def get_side_planes(turn: int):
    """
    4层：红回合、黑回合、九宫掩码、河界掩码
    turn:0红 1黑
    """
    red_turn = np.ones((1, 9, 10), dtype=np.uint8) if turn == 0 else np.zeros((1, 9, 10), dtype=np.uint8)
    blk_turn = np.ones((1, 9, 10), dtype=np.uint8) if turn == 1 else np.zeros((1, 9, 10), dtype=np.uint8)

    palace = np.zeros((1, 9, 10), dtype=np.uint8)
    palace[0, 3:6, 0:3] = 1
    palace[0, 3:6, 7:10] = 1

    river = np.ones((1, 9, 10), dtype=np.uint8)
    return np.concatenate([red_turn, blk_turn, palace, river], axis=0)


# ===================== 象棋棋盘 =====================
class XiangqiBoard:
    def __init__(self):
        self.board = np.zeros((9, 10), dtype=int)
        self.history = []
        self.turn = 0  # 0红 1黑

    def load_fen(self, fen):
        self.board.fill(EMPTY)
        parts = fen.split()
        self.turn = 0 if parts[1] == "w" else 1
        fen_board = parts[0]
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
        self.turn = 1 - self.turn


# ===================== 棋谱解析 =====================
def iccs_move_to_pos(move_str):
    frm, to = move_str.split("-")
    fc = COL_MAP[frm[0]]
    fr = int(frm[1])
    tc = COL_MAP[to[0]]
    tr = int(to[1])
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


# ===================== 数据编码 =====================
EMPTY_STEP = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.float32)


def encode_move(fc, fr, tc, tr):
    step = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.float32)
    step[0, fc, fr] = 1.0
    step[1, tc, tr] = 1.0
    return step


def iccs_games_to_dataset(games, max_games):
    # ------------- 第一步：先统计总步数（不存数据，零内存占用）-------------
    total_steps = 0
    for game in games[:max_games]:
        try:
            total_steps += len(game["moves"])
        except:
            continue

    # ------------- 第二步：预分配内存（核心优化！一次性分配，无碎片）-------------
    # 状态数据: (总步数, 26, 9, 10) float32
    # 🔥 全量终极轻量化类型
    X = np.zeros((total_steps, IN_CHANNELS, 9, 10), dtype=np.uint8)   # 输入 0/1
    y_from = np.zeros(total_steps, dtype=np.int32)                    # 标签 int32
    y_to = np.zeros(total_steps, dtype=np.int32)                      # 标签 int32
    mask_from = np.zeros((total_steps, 90), dtype=np.uint8)           # 掩码 0/1
    mask_to = np.zeros((total_steps, 90), dtype=np.uint8)             # 掩码 0/1

    # 通道索引
    trace_end = STEP_CHANNELS * HISTORY_LEN
    piece_end = trace_end + PIECE_PLANES
    step_idx = 0  # 全局步数指针

    # ------------- 第三步：直接写入预分配数组（不append，不占额外内存）-------------
    for game in tqdm(games[:max_games], desc="生成数据集(低内存版)"):
        board = XiangqiBoard()
        try:
            board.load_fen(game["fen"])
        except:
            continue

        for mv_str in game["moves"]:
            try:
                # 越界直接退出，防止内存爆炸
                if step_idx >= total_steps:
                    break

                move = iccs_move_to_pos(mv_str)
                fc, fr, tc, tr = move

                # 1. 历史轨迹
                hist = board.history
                pad = [EMPTY_STEP] * max(0, HISTORY_LEN - len(hist))
                real_steps = [encode_move(*m) for m in hist[-HISTORY_LEN:]]
                state_trace = np.concatenate(pad + real_steps, axis=0)

                # 2. 直接写入预分配数组（无临时数组，无内存浪费）
                X[step_idx, :trace_end] = state_trace
                X[step_idx, trace_end:piece_end] = board_to_piece_planes(board.board)
                X[step_idx, piece_end:] = get_side_planes(board.turn)

                # 3. 合法掩码
                m_f, m_t = get_legal_mask(board.board, board.turn)
                mask_from[step_idx] = m_f
                mask_to[step_idx] = m_t

                # 4. 标签
                y_from[step_idx] = fc * 10 + fr
                y_to[step_idx] = tc * 10 + tr

                # 步数+1
                step_idx += 1
                board.push(move)
            except:
                continue

    # ------------- 第四步：截断多余空间（完美适配）-------------
    X = X[:step_idx]
    y_from = y_from[:step_idx]
    y_to = y_to[:step_idx]
    mask_from = mask_from[:step_idx]
    mask_to = mask_to[:step_idx]

    return X, y_from, y_to, mask_from, mask_to


# ===================== 缓存工具 =====================
def save_cache(data, path):
    torch.save(data, path)


def load_cache(path):
    return torch.load(path)


# ===================== 模型 =====================
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
        # 输入自动适配26通道
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            activation="gelu",
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,  # 🔥 新增：保存最优loss
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    # 🔥 新增：加载best_loss，兼容旧断点
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(
        f"\n✅ 加载断点成功，从第 {start_epoch} 轮继续训练 | 历史最优Loss: {best_loss:.3f}"
    )
    return start_epoch, best_loss


# ===================== 优雅退出 Ctrl+C =====================
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
    print("🚀 完整版：历史轨迹 + 14棋子盘面 + Side规则平面")

    PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"

    # 自动检测旧缓存不兼容，重建
    if os.path.exists(CACHE_PATH):
        try:
            # 🔥 修正：加载 5 个值（新增掩码）
            X, yf, yt, mfrom, mto = load_cache(CACHE_PATH)
            assert X.shape[1] == IN_CHANNELS
            print("✅ 发现兼容26通道+Legal掩码缓存，快速加载...")
        except Exception:
            print("⚠️  旧缓存不匹配，重新生成完整棋盘数据集...")
            os.remove(CACHE_PATH)
            games = parse_iccs_pgn(PGN_PATH)
            # 🔥 修正：接收 5 个返回值
            X, yf, yt, mfrom, mto = iccs_games_to_dataset(games, max_games=len(games))
            X = torch.tensor(X)
            yf = torch.tensor(yf, dtype=torch.long)
            yt = torch.tensor(yt, dtype=torch.long)
            mfrom = torch.tensor(mfrom)
            mto = torch.tensor(mto)
            # 🔥 修正：保存 5 个值
            save_cache((X, yf, yt, mfrom, mto), CACHE_PATH)
    else:
        print("❌ 首次运行，生成完整棋盘数据集...")
        games = parse_iccs_pgn(PGN_PATH)
        # 🔥 修正：接收 5 个返回值
        X, yf, yt, mfrom, mto = iccs_games_to_dataset(games, max_games=len(games))
        X = torch.tensor(X)
        yf = torch.tensor(yf, dtype=torch.long)
        yt = torch.tensor(yt, dtype=torch.long)
        mfrom = torch.tensor(mfrom)
        mto = torch.tensor(mto)
        # 🔥 修正：保存 5 个值
        save_cache((X, yf, yt, mfrom, mto), CACHE_PATH)
        print(f"✅ 数据集缓存完成，样本数: {len(X)}")

    train_loader = DataLoader(
        TensorDataset(X, yf, yt, mfrom, mto),
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    start_epoch = 0
    best_loss = float("inf")

    # 加载训练进度
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_loss = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )

    print("\n🚀 完整版无限训练启动\n")
    current_epoch = start_epoch

    while True:
        if stop_flag:
            break

        model.train()
        total_loss = 0.0
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}")
        for batch in pbar:
            if stop_flag:
                break
            bx, bf, bt, mfrom, mto = batch
            bx = bx.to(torch.float32)  # 🔥 关键转换
            bx = bx.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)
            mfrom = mfrom.to(device, non_blocking=True)
            mto = mto.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)
            # # 🔥🔥🔥 新增：合法走子约束（核心！强制模型只选合法位置）
            # pred_f = pred_f + torch.where(mfrom.bool(), -torch.inf, 0.0)
            # pred_t = pred_t + torch.where(mto.bool(), -torch.inf, 0.0)
            # 合法=1 → 保留原值
            # 非法=0 → 设为 -inf
            pred_f = torch.where(mfrom == 1, pred_f, -1e9)
            pred_t = torch.where(mto == 1, pred_t, -1e9)

            # 计算Loss（掩码已生效，非法位置概率为0）
            loss = criterion(pred_f, bf) + criterion(pred_t, bt)
            # 防NaN保险
            if not torch.isfinite(loss):
                print(f"⚠️  出现NaN loss，跳过这一批次！")
                optimizer.zero_grad()
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss = running_loss * 0.98 + loss.item() * 0.02
            total_loss += loss.item()
            steps += 1
            avg_loss = total_loss / steps

            pbar.set_postfix(
                {
                    "total_loss": f"{total_loss:.3f}",
                    "run_loss": f"{running_loss:.3f}",
                    "avg": f"{avg_loss:.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {current_epoch} | Loss: {avg_loss:.3f} | LR: {current_lr:.6f}")
        save_checkpoint(
            model, optimizer, scheduler, current_epoch, best_loss, CHECKPOINT_PATH
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 新最优模型 | Best Loss: {best_loss:.3f}\n")

        current_epoch += 1

    save_checkpoint(
        model, optimizer, scheduler, current_epoch - 1, best_loss, CHECKPOINT_PATH
    )
    print("\n🏆 训练已安全停止，下次可直接续训")
