# 只用cpu做训练
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
COL_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
}  # ICCS列映射

# ===================== 1. 原生中国象棋棋盘（支持FEN+ICCS） =====================
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


# ===================== 2. ICCS走法 → 坐标 =====================
def iccs_move_to_pos(move_str):
    move_str = move_str.strip()
    frm, to = move_str.split("-")
    fc = COL_MAP[frm[0]]
    fr = int(frm[1])
    tc = COL_MAP[to[0]]
    tr = int(to[1])
    return (fc, fr, tc, tr)


# ===================== 3. 解析ICCS棋谱文件 =====================
def parse_iccs_pgn(file_path):
    games = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    game_blocks = re.split(r"\n\n(?=\[Game)", content)
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


# ===================== 4. 棋谱 → 模型训练数据 =====================
def iccs_games_to_dataset(games, max_games=2000):
    X, y_from, y_to = [], [], []
    for game in tqdm(games[:max_games], desc="生成训练数据"):
        board = XiangqiBoard()
        board.load_fen(game["fen"])
        moves = game["moves"]
        for mv_str in moves:
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


def _snap_to_tensor(snap):
    t = np.zeros((PIECE_COUNT, 9, 10), np.float32)
    for c in range(9):
        for r in range(10):
            v = snap[c, r]
            if v != EMPTY:
                t[v - 1, c, r] = 1.0
    return t


# ===================== 5. CNN + Transformer 模型（原版不变） =====================
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


# ===================== 6. 主训练（加载你的ICCS棋谱） =====================
if __name__ == "__main__":
    # 👇 把你的ICCS棋谱文件名改这里！
    PGN_PATH = "./iccs_lib/ICCS-41743/WXF-41743games.pgns"

    print("加载ICCS棋谱...")
    games = parse_iccs_pgn(PGN_PATH)
    print(f"共加载 {len(games)} 局棋谱")

    print("构建训练集...")
    X, yf, yt = iccs_games_to_dataset(games, max_games=5000)
    print(f"训练样本数：{len(X)}")

    # 数据加载
    X = torch.tensor(X)
    yf = torch.tensor(yf, dtype=torch.long)
    yt = torch.tensor(yt, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, yf, yt), batch_size=16, shuffle=True)

    # 训练配置
    device = torch.device("cpu")
    model = HybridXiangqiModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 CNN+Transformer 象棋模型...")
    for epoch in range(12):
        model.train()
        loss_sum = 0
        for bx, bf, bt in tqdm(loader):
            bx, bf, bt = bx.to(device), bf.to(device), bt.to(device)
            opt.zero_grad()
            pf, pt = model(bx)
            loss = criterion(pf, bf) + criterion(pt, bt)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1} | Loss: {loss_sum/len(loader):.3f}")

    torch.save(model.state_dict(), "xiangqi_iccs_cnn_transformer.pth")
    print("训练完成！模型已保存")
