# 优化版：动态学习率 + 快速收敛，比v0版增加自动调节下降梯度
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


# 搞定。自动学习率调节有三层机制：
# ┌─ ReduceLROnPlateau (PyTorch内置)
# │   patience=2: 连续2轮Loss不降 → LR×0.5
# │   cooldown=1: 降完后等1轮再监控
# │
# ├─ 手动加强 (自定义)
# │   patience=5: 连续5轮不降(即使ReduceLROnPlateau没触发)
# │   → 强制 LR×0.5（防止调度器反应慢）
# │
# └─ 自动停止
#     LR < 1e-6 且连续10轮不降 → 自动结束训练

import numpy as np
import torch
import torch.nn as nn
import re
import os
import signal
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
    "K": RED_K, "R": RED_R, "N": RED_N, "H": RED_N,  # H=红马(ICCS别名)
    "C": RED_C, "E": RED_E, "B": RED_E,               # B=红象(标准FEN)
    "A": RED_A, "P": RED_P,
    "k": BLK_K, "r": BLK_R, "n": BLK_N, "h": BLK_N,
    "c": BLK_C, "e": BLK_E, "b": BLK_E,
    "a": BLK_A, "p": BLK_P,
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
    """
    伪合法走法（不检测将军）
    FEN坐标系：行0=黑方底线(上方)，行9=红方底线(下方)
    """
    pid = board[c0, r0]
    if pid == EMPTY:
        return []
    moves = []

    # ═══ 九宫常量（修复：红方九宫在底部7-9行，黑方在顶部0-2行） ═══
    def RED_PALACE(c, r):
        return 3 <= c <= 5 and 7 <= r <= 9

    def BLK_PALACE(c, r):
        return 3 <= c <= 5 and 0 <= r <= 2

    # 将/帅
    if pid in (RED_K, BLK_K):
        for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            c1, r1 = c0 + dc, r0 + dr
            if not in_border(c1, r1):
                continue
            if is_red(pid) and not RED_PALACE(c1, r1):
                continue
            if is_black(pid) and not BLK_PALACE(c1, r1):
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
            if is_red(pid) and not RED_PALACE(c1, r1):
                continue
            if is_black(pid) and not BLK_PALACE(c1, r1):
                continue
            if (is_red(pid) and board[c1, r1] in RED_PIECES) or \
               (is_black(pid) and board[c1, r1] in BLK_PIECES):
                continue
            moves.append((c1, r1))

    # 象（修复：红方半场=行5-9，黑方半场=行0-4）
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
            # ❌ 旧: is_red and r1>=5 → continue (跳过合法走法!)
            # ✅ 新: 红象不能过河(行<=4为黑方)，黑象不能过河(行>=5为红方)
            if is_red(pid) and r1 <= 4:   # 红象试图过河到黑方 → 非法
                continue
            if is_black(pid) and r1 >= 5:  # 黑象试图过河到红方 → 非法
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
            dirs = [(0, -1)]          # 红兵向上（行号减小）
            if r0 <= 4:                # 过河后可左右
                dirs += [(1, 0), (-1, 0)]
        else:
            dirs = [(0, 1)]           # 黑兵向下（行号增大）
            if r0 >= 5:                # 过河后可左右
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
    """
    生成 legal from/to 掩码 (90,)
    合法=0.0，非法=-1e4（exp(-1e4)=0，且 float16 可表示）
    """
    FILL = -1e4
    from_mask = np.full(90, FILL, dtype=np.float32)
    to_mask = np.full(90, FILL, dtype=np.float32)
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
        fr = 9 - int(frm[1])        # ICCS 行(0=红方) → FEN 行(0=黑方)
        tc = COL_MAP[to[0].upper()]
        tr = 9 - int(to[1])         # ICCS 行(0=红方) → FEN 行(0=黑方)
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

# 通道范围常量（供 reconstruct_state 使用）
TRACE_END = STEP_CHANNELS * HISTORY_LEN            # 8
PIECE_END = TRACE_END + PIECE_PLANES                # 22 (8+14)


def encode_move(fc, fr, tc, tr):
    step = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)
    step[0, fc, fr] = 1
    step[1, tc, tr] = 1
    return step


def reconstruct_state(board_arr, history_moves, turn):
    """
    从紧凑表示重建完整26通道状态 (26,9,10) uint8
    单次预分配 + 原地填充 → 零中间数组开销
    """
    state = np.zeros((IN_CHANNELS, 9, 10), dtype=np.uint8)

    # 历史轨迹 (0..7) — 末尾对齐，前面留空（与旧版 pad+real_steps 一致）
    pad_start = HISTORY_LEN - len(history_moves)
    for i, m in enumerate(history_moves):
        if i >= HISTORY_LEN:
            break
        ch = (pad_start + i) * 2
        fc, fr, tc, tr = m
        state[ch, fc, fr] = 1
        state[ch + 1, tc, tr] = 1

    # 棋子平面 (8..21) — 用 np.equal(out=) 避免中间数组
    for idx, pid in enumerate(ALL_PIECE_IDS):
        np.equal(board_arr, pid, out=state[TRACE_END + idx])

    # 规则平面 (22..25) — 直接写入
    if turn == 0:
        state[PIECE_END] = 1       # 红回合
    else:
        state[PIECE_END + 1] = 1   # 黑回合
    state[PIECE_END + 2, 3:6, 0:3] = 1   # 红九宫
    state[PIECE_END + 2, 3:6, 7:10] = 1  # 黑九宫
    state[PIECE_END + 3, :, :5] = 1      # 红半场

    return state


def iccs_games_to_dataset_compact(games, max_games):
    """
    紧凑版：只存棋盘+历史走法+回合，不存完整26通道状态
    返回 (boards, histories, turns, yf, yt, mf, mt)
    单样本从 2340B 降到 107B（不含掩码）
    """
    boards, histories, turns = [], [], []
    y_from, y_to, mask_from, mask_to = [], [], [], []

    for game in tqdm(games[:max_games], desc="生成数据集(紧凑版)"):
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

                # 存棋盘快照 (9,10) uint8
                boards.append(board.board.copy())

                # 存历史走法 (HISTORY_LEN, 4) 用-1填充空白
                hist = board.history
                padded = np.full((HISTORY_LEN, 4), -1, dtype=np.int8)
                for i, m in enumerate(hist[-HISTORY_LEN:]):
                    padded[i] = m
                histories.append(padded)

                # 存回合
                turns.append(board.turn)

                # 标签
                y_from.append(fc * 10 + fr)
                y_to.append(tc * 10 + tr)

                # 合法掩码
                m_f, m_t = get_legal_mask(board.board, board.turn)
                mask_from.append(m_f)
                mask_to.append(m_t)

                board.push(move)
            except Exception:
                continue

    return (
        "CACHE_v2",                                  # 版本号（旧缓存是7项，自动失效）
        np.array(boards, dtype=np.uint8),            # (N,9,10) 1B
        np.array(histories, dtype=np.int8),          # (N,HISTORY_LEN,4) 1B
        np.array(turns, dtype=np.uint8),             # (N,) 1B
        np.array(y_from, dtype=np.uint8),            # (N,) 1B
        np.array(y_to, dtype=np.uint8),              # (N,) 1B
        np.array(mask_from, dtype=np.float32),       # (N,90) 4B（不用float16避免溢出）
        np.array(mask_to, dtype=np.float32),         # (N,90) 4B
    )


class CompactXiangqiDataset(Dataset):
    """
    紧凑象棋数据集：存棋盘+历史+回合，__getitem__ 按需重建26通道状态
    内存占用减少约 5.8 倍
    """

    def __init__(self, boards, histories, turns, yf, yt, mf, mt):
        self.boards = boards           # (N,9,10) uint8
        self.histories = histories     # (N,4,4) int8
        self.turns = turns             # (N,) uint8
        self.yf = yf                   # (N,) uint8
        self.yt = yt                   # (N,) uint8
        self.mf = mf                   # (N,90) float32
        self.mt = mt                   # (N,90) float32

    def __len__(self):
        return len(self.yf)

    def __getitem__(self, idx):
        # 按需重建26通道状态（~0.1ms/样本，可接受）
        board = self.boards[idx]
        history = self.histories[idx]
        turn = int(self.turns[idx])

        # 过滤掉-1填充的历史
        valid_history = [tuple(m) for m in history if m[0] >= 0]

        state = reconstruct_state(board, valid_history, turn)

        return (
            torch.from_numpy(state),                        # uint8
            torch.tensor(self.yf[idx], dtype=torch.long),
            torch.tensor(self.yt[idx], dtype=torch.long),
            torch.from_numpy(self.mf[idx]),    # 已在加载时转为float32
            torch.from_numpy(self.mt[idx]),
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
def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path,
                    lr_patience_counter=0, lr_best_loss=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }
    if lr_best_loss is not None:
        state["lr_patience_counter"] = lr_patience_counter
        state["lr_best_loss"] = lr_best_loss
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "num_bad_epochs" in checkpoint.get("scheduler_state_dict", {}):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    lr_patience_counter = checkpoint.get("lr_patience_counter", 0)
    lr_best_loss = checkpoint.get("lr_best_loss", float("inf"))
    print(f"\n✅ 加载断点成功，从第 {start_epoch} 轮继续训练 | 历史最优Loss: {best_loss:.3f}")
    return start_epoch, best_loss, lr_patience_counter, lr_best_loss


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

    # 加载/生成紧凑数据集（版本号校验，旧缓存自动废弃）
    if os.path.exists(CACHE_PATH):
        try:
            data = load_cache(CACHE_PATH)
            if isinstance(data, (list, tuple)) and len(data) == 8 and data[0] == "CACHE_v2":
                _, boards, histories, turns, yf, yt, mf, mt = data
            else:
                raise ValueError(f"旧缓存格式(len={len(data) if hasattr(data,'__len__') else '?'})")
            # 确保内存中为float32
            mf = np.asarray(mf, dtype=np.float32)
            mt = np.asarray(mt, dtype=np.float32)
            print(f"✅ 发现兼容缓存，快速加载... 样本数: {len(yf)}")
            print(f"   紧凑存储: {boards.nbytes/1e9:.1f}GB (vs 全状态 {(boards.nbytes * IN_CHANNELS * 9 * 10 / (9*10))/1e9:.1f}GB)")
        except Exception as e:
            print(f"⚠️  缓存不兼容 ({e})，重新生成紧凑数据集...")
            if os.path.exists(CACHE_PATH):
                os.remove(CACHE_PATH)
            games = parse_iccs_pgn(PGN_PATH)
            _, boards, histories, turns, yf, yt, mf, mt = iccs_games_to_dataset_compact(games, max_games=len(games))
            mf = np.asarray(mf, dtype=np.float32)
            mt = np.asarray(mt, dtype=np.float32)
            save_cache(("CACHE_v2", boards, histories, turns, yf, yt, mf, mt), CACHE_PATH)
            print(f"✅ 数据集重建完成，样本数: {len(yf)}")
    else:
        print("❌ 首次运行，生成训练数据...")
        games = parse_iccs_pgn(PGN_PATH)
        _, boards, histories, turns, yf, yt, mf, mt = iccs_games_to_dataset_compact(games, max_games=len(games))
        mf = np.asarray(mf, dtype=np.float32)
        mt = np.asarray(mt, dtype=np.float32)
        save_cache(("CACHE_v2", boards, histories, turns, yf, yt, mf, mt), CACHE_PATH)
        print(f"✅ 紧凑数据集缓存完成，样本数: {len(yf)}")

    # ════════════════════════════════════════════════
    # 🔍 数据集完整性诊断
    # ════════════════════════════════════════════════
    print("\n🔍 数据集诊断...")
    n_check = min(5000, len(yf))
    bad_from = 0; bad_to = 0; all_inf_from = 0; all_inf_to = 0
    empty_board = 0
    for i in range(n_check):
        # 检查棋盘棋子数
        if np.count_nonzero(boards[i]) < 2:
            empty_board += 1
        # 检查标签是否在合法掩码内
        if mf[i, yf[i]] != 0.0:
            bad_from += 1
        if mt[i, yt[i]] != 0.0:
            bad_to += 1
        # 检查全非法掩码
        if np.all(mf[i] < -1e8):
            all_inf_from += 1
        if np.all(mt[i] < -1e8):
            all_inf_to += 1
    print(f"  棋盘: {n_check - empty_board}/{n_check} 非空")
    print(f"  from标签匹配mask: {n_check - bad_from}/{n_check} ✅" if bad_from == 0 else f"  ❌ from标签非法: {bad_from}/{n_check}")
    print(f"  to标签匹配mask:   {n_check - bad_to}/{n_check} ✅" if bad_to == 0 else f"  ❌ to标签非法:   {bad_to}/{n_check}")
    print(f"  全非法from mask: {all_inf_from}  全非法to mask: {all_inf_to}")
    if bad_from > 0 or bad_to > 0:
        print("  ⚠️  标签与mask不匹配，请检查 iccs_move_to_pos 行反转 或 FEN_PIECE 别名")
        # 打印第一个出错样本的详情
        for i in range(min(5, n_check)):
            if mf[i, yf[i]] != 0.0 or mt[i, yt[i]] != 0.0:
                fc, fr = yf[i] // 10, yf[i] % 10
                tc, tr = yt[i] // 10, yt[i] % 10
                board_str = str(boards[i, :, :].T)  # 转置便于查看
                print(f"\n  样本{i}: turn={turns[i]}")
                print(f"    from=({fc},{fr}) mask_val={mf[i, yf[i]]}  board_val={boards[i, fc, fr]}")
                print(f"    to=({tc},{tr})   mask_val={mt[i, yt[i]]}  board_val={boards[i, tc, tr]}")
                print(f"    棋盘非零位置: {np.argwhere(boards[i] != 0)[:5].tolist()}")
                break

    # 🔍 迷你前向传播测试
    print("\n🔍 迷你前向传播测试...")
    test_dataset = CompactXiangqiDataset(boards[:16], histories[:16], turns[:16], yf[:16], yt[:16], mf[:16], mt[:16])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_model = HybridXiangqiModel()
    test_model.eval()
    with torch.no_grad():
        for bx, bf, bt, bm_f, bm_t in test_loader:
            pf, pt = test_model(bx.float())
            pf = pf + bm_f
            pt = pt + bm_t
            lf = nn.CrossEntropyLoss()(pf, bf)
            lt = nn.CrossEntropyLoss()(pt, bt)
            print(f"  pred_f: range=[{pf.min():.4f}, {pf.max():.4f}] nan={torch.isnan(pf).any()}")
            print(f"  pred_t: range=[{pt.min():.4f}, {pt.max():.4f}] nan={torch.isnan(pt).any()}")
            print(f"  loss_f={lf:.4f}  loss_t={lt:.4f}  total={lf+lt:.4f}")
            if torch.isnan(lf) or torch.isinf(lf) or torch.isnan(lt) or torch.isinf(lt):
                print("  ❌ Loss异常! 检查标签是否越界或mask全-inf")
            else:
                print("  ✅ 前向传播正常")
            break
    del test_model, test_dataset, test_loader
    print()

    train_dataset = CompactXiangqiDataset(boards, histories, turns, yf, yt, mf, mt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # 模型初始化
    model = HybridXiangqiModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
        cooldown=1, min_lr=5e-6, verbose=True
    )
    criterion = nn.CrossEntropyLoss().to(device)
    start_epoch = 0
    best_loss = float("inf")
    # 自动学习率调节状态
    lr_patience_counter = 0
    lr_best_loss = float("inf")

    # 加载断点（自动跳过不兼容的旧断点）
    if os.path.exists(CHECKPOINT_PATH):
        try:
            start_epoch, best_loss, lr_patience_counter, lr_best_loss = load_checkpoint(
                CHECKPOINT_PATH, model, optimizer, scheduler, device
            )
        except RuntimeError as e:
            print(f"⚠️  断点不兼容（通道数从12→26），删除旧断点重新训练")
            print(f"   原因: {e}")
            os.remove(CHECKPOINT_PATH)
            if os.path.exists(BEST_MODEL_PATH):
                os.remove(BEST_MODEL_PATH)
            start_epoch = 0
            best_loss = float("inf")

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
            bx = bx.float().to(device)          # uint8 → float32 → MPS
            bf = bf.long().to(device)           # uint8 → long → MPS
            bt = bt.long().to(device)           # uint8 → long → MPS
            bm_f = bm_f.to(device)              # float32 → MPS
            bm_t = bm_t.to(device)

            optimizer.zero_grad()
            pred_f, pred_t = model(bx)

            # 🔥 合法走法掩码：非法位置 logit 压低，softmax 后概率≈0
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

        # ── 自动学习率调节 ──
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 如果 Loss 连续不降，自动做更激进的衰减
        if avg_loss < lr_best_loss * 0.999:
            lr_best_loss = avg_loss
            lr_patience_counter = 0
        else:
            lr_patience_counter += 1

        # 连续5轮不降 → 手动砍半LR（比ReduceLROnPlateau更快响应）
        if lr_patience_counter >= 5 and current_lr > 1e-5:
            old_lr = current_lr
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            current_lr = optimizer.param_groups[0]["lr"]
            lr_patience_counter = 0
            lr_best_loss = float("inf")
            print(f"⚡ 自动降LR: {old_lr:.6f} → {current_lr:.6f}")

        # LR 过低时自动停止（防止浪费算力）
        if current_lr < 1e-6 and lr_patience_counter >= 10:
            print("\n🛑 LR 过低且 Loss 不再下降，自动停止训练")
            break

        print(f"✅ Epoch {current_epoch} | Loss: {avg_loss:.3f} | LR: {current_lr:.6f}")
        save_checkpoint(model, optimizer, scheduler, current_epoch, best_loss,
                        CHECKPOINT_PATH, lr_patience_counter, lr_best_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 新最优模型 | Best Loss: {best_loss:.3f}\n")

        current_epoch += 1

    save_checkpoint(model, optimizer, scheduler, current_epoch - 1, best_loss,
                    CHECKPOINT_PATH, lr_patience_counter, lr_best_loss)
    print("\n🏆 训练已安全停止，下次可直接续训")
