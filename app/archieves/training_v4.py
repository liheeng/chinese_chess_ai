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

# ===================== 全局配置 =====================
HISTORY_LEN = 4
STEP_CHANNELS = 2
IN_CHANNELS = STEP_CHANNELS * HISTORY_LEN
COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}

# 文件路径（仅1个断点，不占空间）
CACHE_PATH = "xiangqi_dataset_cache.pt"
CHECKPOINT_PATH = "ckpt_latest.pth"
BEST_MODEL_PATH = "xiangqi_best.pth"

# 棋子定义
EMPTY = 0
RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P = 1, 2, 3, 4, 5, 6, 7
BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P = 8, 9, 10, 11, 12, 13, 14

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


# ===================== 象棋棋盘 =====================
class XiangqiBoard:
    def __init__(self):
        self.board = np.zeros((9, 10), dtype=int)
        self.history = []

    def load_fen(self, fen):
        self.board.fill(EMPTY)
        fen_board = fen.split()[0]
        rows = fen_board.split("/")
        for r_idx, row in enumerate(rows):
            c_idx = 0
            for ch in row:
                if ch.isdigit():
                    c_idx += int(ch)
                    continue
                else:
                    if ch in FEN_PIECE and c_idx < 9:
                        self.board[c_idx, r_idx] = FEN_PIECE[ch]
                c_idx += 1

    def push(self, move):
        fc, fr, tc, tr = move
        self.history.append((fc, fr, tc, tr))
        self.board[tc, tr] = self.board[fc, fr]
        self.board[fc, fr] = EMPTY


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
    X, y_from, y_to = [], [], []
    target_shape = (IN_CHANNELS, 9, 10)

    for game in tqdm(games[:max_games], desc="生成训练数据"):
        board = XiangqiBoard()
        try:
            board.load_fen(game["fen"])
        except Exception:
            continue

        # 🔥 修复：补全括号
        for mv_str in game["moves"]:
            try:
                move = iccs_move_to_pos(mv_str)
                fc, fr, tc, tr = move

                hist = board.history
                pad = [EMPTY_STEP] * max(0, HISTORY_LEN - len(hist))
                real_steps = [encode_move(*m) for m in hist[-HISTORY_LEN:]]
                state = np.concatenate(pad + real_steps, axis=0)

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
    # with gzip.open(path, "wb") as f:
    torch.save(data, path)


def load_cache(path):
    # with gzip.open(path, "rb") as f:
    return torch.load(path, weights_only=True)


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
    # 🔥 升级：nhead=4，CNN加一层，最强降Loss
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            # 🔥 新增一层CNN，小幅提升空间特征提取
            nn.Conv2d(64, d_model, 3, padding=1), nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
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
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"\n✅ 加载断点成功，从第 {start_epoch} 轮继续训练")
    return start_epoch


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
    print("🚀 优化版：动态学习率 + 快速收敛")

    PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"

    # 加载数据集缓存
    if os.path.exists(CACHE_PATH):
        print("✅ 发现数据集缓存，快速加载...")
        X, yf, yt = load_cache(CACHE_PATH)
    else:
        print("❌ 首次运行，生成训练数据...")
        games = parse_iccs_pgn(PGN_PATH)
        X, yf, yt = iccs_games_to_dataset(games, max_games=len(games))
        X = torch.tensor(X)
        yf = torch.tensor(yf, dtype=torch.long)
        yt = torch.tensor(yt, dtype=torch.long)
        save_cache((X, yf, yt), CACHE_PATH)
        print(f"✅ 数据集缓存完成，样本数: {len(X)}")

    # 优化1：BatchSize=256 收敛更快
    # MPS 上 num_workers=0 避免子进程内存开销
    train_loader = DataLoader(
        TensorDataset(X, yf, yt),
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # 模型初始化
    model = HybridXiangqiModel().to(device)
    # 优化2：AdamW + 权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    # 优化3：余弦学习率调度（核心加速）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss().to(device)
    start_epoch = 0
    best_loss = float("inf")

    # 加载训练进度
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )

    print("\n🚀 无限训练模式启动\n")
    current_epoch = start_epoch

    while True:
        if stop_flag:
            break

        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {current_epoch}"):
            if stop_flag:
                break
            bx, bf, bt = batch
            bx = bx.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)
            loss = criterion(pred_f, bf) + criterion(pred_t, bt)
            loss.backward()

            # 优化4：梯度裁剪，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # 更新学习率
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {current_epoch} | Loss: {avg_loss:.3f} | LR: {current_lr:.6f}")
        save_checkpoint(model, optimizer, scheduler, current_epoch, CHECKPOINT_PATH)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 新最优模型 | Best Loss: {best_loss:.3f}\n")

        current_epoch += 1

    # 最终保存
    save_checkpoint(model, optimizer, scheduler, current_epoch - 1, CHECKPOINT_PATH)
    print("\n🏆 训练已安全停止，下次可直接续训")
