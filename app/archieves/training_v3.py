# 去棋谱张量冗余
# 棋谱：41743盘，样本数350万
# CNN：3 层
# Transformer：nhead=2
# 12轮测试结果： loss逐步收敛，但下降幅度越来越小，效果一般
# Epoch  1 | Loss: 6.792
# Epoch  2 | Loss: 6.446
# Epoch  3 | Loss: 6.379
# Epoch  4 | Loss: 6.343
# Epoch  5 | Loss: 6.320
# Epoch  6 | Loss: 6.304
# Epoch  7 | Loss: 6.290
# Epoch  8 | Loss: 6.281
# Epoch  9 | Loss: 6.271
# Epoch 10 | Loss: 6.265
# Epoch 11 | Loss: 6.258
# Epoch 12 | Loss: 6.253
# ---------------------------------
import numpy as np
import torch
import torch.nn as nn
import re
import gzip
import os
# from typing import Any, BinaryIO, cast
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# ===================== 全局配置 =====================
HISTORY_LEN = 4
STEP_CHANNELS = 2
IN_CHANNELS = STEP_CHANNELS * HISTORY_LEN
COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}
XIANGQI_DATASET_CACHE_PATH = "./xiangqi_dataset_cache.pt.gz"

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
    with gzip.open(path, "wb") as f:
        torch.save(data, f)


def load_cache(path):
    with gzip.open(path, "rb") as f:
        return torch.load(f)


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
    def __init__(self, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.from_head = nn.Linear(d_model, 90)
        self.to_head = nn.Linear(d_model, 90)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.pos_enc(x)
        x = self.transformer(x).permute(1, 0, 2)
        return self.from_head(x).mean(1), self.to_head(x).mean(1)


# ===================== 训练主程序 =====================
if __name__ == "__main__":
    # ✅ 修复：正确的 MPS 检测写法
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")

    PGN_PATH = "./iccs_lib/ICCS-41743/WXF-41743games.pgns"

    # 缓存逻辑
    if os.path.exists(XIANGQI_DATASET_CACHE_PATH):
        print("✅ 发现缓存文件，直接加载数据集...")
        X, yf, yt = load_cache(XIANGQI_DATASET_CACHE_PATH)
    else:
        print("❌ 未找到缓存，首次生成训练数据...")
        games = parse_iccs_pgn(PGN_PATH)
        print(f"✅ 总棋谱数: {len(games)}")
        X, yf, yt = iccs_games_to_dataset(games, max_games=len(games))
        X = torch.tensor(X)
        yf = torch.tensor(yf, dtype=torch.long)
        yt = torch.tensor(yt, dtype=torch.long)
        save_cache((X, yf, yt), XIANGQI_DATASET_CACHE_PATH)
        print(f"✅ 数据集已压缩缓存到: {XIANGQI_DATASET_CACHE_PATH}")

    print(f"✅ 训练样本: {len(X)}")
    print(f"✅ 数据形状: {X.shape}")

    train_loader = DataLoader(
        TensorDataset(X, yf, yt),
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = HybridXiangqiModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    print("\n🚀 开始训练...")
    for epoch in range(12):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, bf, bt = batch
            bx = bx.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)
            loss = criterion(pred_f, bf) + criterion(pred_t, bt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.3f}")

    torch.save(model.state_dict(), "xiangqi_final.pth")
    print("\n🏆 训练完成！模型已保存")
