# cpu+gpu混合训练，cpu做IO，gpu做cnn和transformer
import numpy as np
import torch
import torch.nn as nn
import re
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# ===================== 全局配置 =====================
PIECE_COUNT = 14  # 红7+黑7
HISTORY_LEN = 4  # 输入历史步数
IN_CHANNELS = PIECE_COUNT * HISTORY_LEN  # 模型输入通道=56
COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}

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


# ===================== 原生象棋棋盘 =====================
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

    def snapshot(self):
        return self.board.copy()

    def push(self, move):
        fc, fr, tc, tr = move
        self.history.append(self.snapshot())
        self.board[tc, tr] = self.board[fc, fr]
        self.board[fc, fr] = EMPTY


# ===================== ICCS 棋谱解析 =====================
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


# ===================== 数据编码（CPU 异步处理） =====================
def _snap_to_tensor(snap):
    t = np.zeros((PIECE_COUNT, 9, 10), np.float32)
    for c in range(9):
        for r in range(10):
            v = snap[c, r]
            if v != EMPTY:
                t[v - 1, c, r] = 1.0
    return t


def iccs_games_to_dataset(games, max_games=5000):
    X, y_from, y_to = [], [], []
    for game in tqdm(games[:max_games], desc="CPU 解析棋谱"):
        board = XiangqiBoard()
        board.load_fen(game["fen"])
        for mv_str in game["moves"]:
            try:
                move = iccs_move_to_pos(mv_str)
                if len(board.history) >= HISTORY_LEN:
                    hist = board.history[-HISTORY_LEN:]
                    pad = [np.zeros((9, 10))] * (HISTORY_LEN - len(hist))
                    snaps = pad + hist
                    tens = np.concatenate([_snap_to_tensor(s) for s in snaps], axis=0)
                    fid = move[0] * 10 + move[1]
                    tid = move[2] * 10 + move[3]
                    X.append(tens)
                    y_from.append(fid)
                    y_to.append(tid)
                board.push(move)
            except Exception:
                continue
    return np.array(X, np.float32), np.array(y_from), np.array(y_to)


# ===================== CNN + Transformer 模型 =====================
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


# ===================== 🔥 M5 Max 混合训练（CPU+GPU，无AMP纯兼容） =====================
if __name__ == "__main__":
    # 0. 自动检测 M5 Max GPU(MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 混合训练模式：GPU(MPS)={device.type} | CPU=数据加载/解析")

    # 1. 加载棋谱（CPU 处理）
    # PGN_PATH = "chinese-chess-pgn.iccs"
    PGN_PATH = "./iccs_lib/ICCS-41743/WXF-41743games.pgns"
    games = parse_iccs_pgn(PGN_PATH)
    print(f"✅ 加载完成：{len(games)} 局棋谱")

    # 2. 构建数据集（CPU 异步预处理）
    X, yf, yt = iccs_games_to_dataset(games, max_games=5000)
    print(f"✅ 训练样本：{len(X)}")

    # 3. 数据加载器（CPU 并行加载 + GPU 异步传输）
    X = torch.tensor(X)
    yf = torch.tensor(yf, dtype=torch.long)
    yt = torch.tensor(yt, dtype=torch.long)
    dataset = TensorDataset(X, yf, yt)

    # 🚀 M5 Max 优化：CPU 异步加载，统一内存零拷贝
    train_loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )

    # 4. 模型 + 优化器（GPU 运算）
    model = HybridXiangqiModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    print("\n🚀 开始 CPU+GPU 混合训练...")
    for epoch in range(12):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, bf, bt = batch
            # 异步数据传输（统一内存，无拷贝开销）
            bx = bx.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 🔥 直接前向传播（MPS GPU 全速运行）
            pred_f, pred_t = model(bx)
            loss = criterion(pred_f, bf) + criterion(pred_t, bt)

            # 反向传播 + 参数更新（GPU 核心计算）
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.3f}")

    # 保存模型
    torch.save(model.state_dict(), "xiangqi_m5max_hybrid.pth")
    print("\n🏆 训练完成！M5 Max 混合训练模型已保存")
