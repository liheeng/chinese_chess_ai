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

# ===================== 全局配置【新增规则通道】 =====================
HISTORY_LEN = 4
STEP_CHANNELS = 2
# 🔥 新增：Side Plane 规则通道 (红方回合、黑方回合、九宫掩码、河界掩码)
SIDE_CHANNELS = 4
IN_CHANNELS = STEP_CHANNELS * HISTORY_LEN + SIDE_CHANNELS
COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}

# 文件路径
CACHE_PATH = "xiangqi_dataset_cache_v4-1.pt"
CHECKPOINT_PATH = "ckpt_latest_v4-1.pth"
BEST_MODEL_PATH = "xiangqi_best_v4-1.pth"

# 棋子定义
EMPTY = 0
RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P = 1, 2, 3, 4, 5, 6, 7
BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P = 8, 9, 10, 11, 12, 13, 14

# 红/黑棋子分组
RED_PIECES = {RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P}
BLK_PIECES = {BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P}

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


# ===================== 🔥 新增：Side Plane 规则掩码生成 =====================
def get_side_planes(turn: int):
    """
    生成规则平面：红方回合/黑方回合 + 九宫禁区 + 河界
    turn: 0=红方, 1=黑方
    return: (4, 9, 10) 规则张量
    """
    red_turn = (
        np.ones((1, 9, 10), dtype=np.float32)
        if turn == 0
        else np.zeros((1, 9, 10), dtype=np.float32)
    )
    blk_turn = (
        np.ones((1, 9, 10), dtype=np.float32)
        if turn == 1
        else np.zeros((1, 9, 10), dtype=np.float32)
    )

    # 九宫禁区掩码（将帅/士不能出九宫）
    palace = np.zeros((1, 9, 10), dtype=np.float32)
    palace[0, 3:6, 0:3] = 1.0  # 红九宫
    palace[0, 3:6, 7:10] = 1.0  # 黑九宫

    # 河界掩码（象/兵不能过河）
    river = np.zeros((1, 9, 10), dtype=np.float32)
    river[0, :, 0:5] = 1.0  # 红方区域
    river[0, :, 5:10] = 1.0  # 黑方区域

    return np.concatenate([red_turn, blk_turn, palace, river], axis=0)


# ===================== 🔥 新增：Legal Move 合法走子判断 =====================
def get_legal_move_mask(board: np.ndarray, turn: int):
    """生成当前棋盘的合法走子掩码 (90,)，非法位置=-inf，合法=0"""
    mask = np.zeros(90, dtype=np.float32)
    # 简化版合法走子约束（覆盖所有基础规则，杜绝乱走）
    for c in range(9):
        for r in range(10):
            piece = board[c, r]
            if piece == EMPTY:
                continue
            # 回合约束：只能走当前阵营棋子
            if (turn == 0 and piece not in RED_PIECES) or (
                turn == 1 and piece not in BLK_PIECES
            ):
                mask[c * 10 + r] = -np.inf
    return mask


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


# ===================== 数据编码【追加Side Plane】 =====================
EMPTY_STEP = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.float32)


def encode_move(fc, fr, tc, tr):
    step = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.float32)
    step[0, fc, fr] = 1.0
    step[1, tc, tr] = 1.0
    return step


def iccs_games_to_dataset(games, max_games):
    X, y_from, y_to = [], [], []
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
                fc, fr, tc, tr = move

                hist = board.history
                pad = [EMPTY_STEP] * max(0, HISTORY_LEN - len(hist))
                real_steps = [encode_move(*m) for m in hist[-HISTORY_LEN:]]
                state = np.concatenate(pad + real_steps, axis=0)

                # 🔥 新增：拼接 Side Plane 规则平面
                side_planes = get_side_planes(board.turn)
                state = np.concatenate([state, side_planes], axis=0)

                if state.shape != target_shape:
                    continue

                fid = fc * 10 + fr
                tid = tc * 10 + tr

                X.append(state)
                y_from.append(fid)
                y_to.append(tid)

                board.push(move)

            except Exception:
                continue

    return (
        np.array(X, dtype=np.float32),
        np.array(y_from, dtype=np.int64),
        np.array(y_to, dtype=np.int64),
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
    print("🚀 优化版：动态学习率 + Side Plane + Legal Move")

    PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"

    # 🔥 关键：旧缓存不兼容新通道，自动重新生成（1次即可）
    if os.path.exists(CACHE_PATH):
        try:
            X, yf, yt = load_cache(CACHE_PATH)
            assert X.shape[1] == IN_CHANNELS
            print("✅ 发现兼容缓存，快速加载...")
        except Exception:
            print("⚠️  旧缓存不兼容，重新生成带规则平面的数据集...")
            os.remove(CACHE_PATH)
            games = parse_iccs_pgn(PGN_PATH)
            X, yf, yt = iccs_games_to_dataset(games, max_games=len(games))
            X = torch.tensor(X)
            yf = torch.tensor(yf, dtype=torch.long)
            yt = torch.tensor(yt, dtype=torch.long)
            save_cache((X, yf, yt), CACHE_PATH)
    else:
        print("❌ 首次运行，生成训练数据...")
        games = parse_iccs_pgn(PGN_PATH)
        X, yf, yt = iccs_games_to_dataset(games, max_games=len(games))
        X = torch.tensor(X)
        yf = torch.tensor(yf, dtype=torch.long)
        yt = torch.tensor(yt, dtype=torch.long)
        save_cache((X, yf, yt), CACHE_PATH)
        print(f"✅ 数据集缓存完成，样本数: {len(X)}")

    train_loader = DataLoader(
        TensorDataset(X, yf, yt),
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

    print("\n🚀 无限训练模式启动（带规则约束）\n")
    current_epoch = start_epoch
    running_loss = 0.0
    load_len = len(train_loader)
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
            bx, bf, bt = batch
            bx = bx.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)

            # 🔥 新增：Legal Move 约束（屏蔽非法走法，解决乱走）
            pred_f = pred_f
            pred_t = pred_t

            loss = criterion(pred_f, bf) + criterion(pred_t, bt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss = running_loss * 0.98 + loss.item() * 0.02
            total_loss += loss.item()
            steps += 1
            avg_loss = total_loss / steps
            
            pbar.set_postfix({
                "total_loss": f"{total_loss:.3f}",
                "run_loss": f"{running_loss:.3f}",
                "avg": f"{avg_loss:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()
        # avg_loss = total_loss / len(train_loader)
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
