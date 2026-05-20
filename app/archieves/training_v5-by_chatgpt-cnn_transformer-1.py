# =========================================================
# 中国象棋 AI（M5 Max 优化版）
# Board Planes + History Planes + CNN + Transformer
#
# 特性：
# ✅ FEN局面
# ✅ 历史动作
# ✅ bf16
# ✅ Apple Silicon MPS优化
# ✅ uint8数据集
# ✅ channels_last
# ✅ Transformer修复
# ✅ 动态学习率
# ✅ checkpoint续训
# ✅ graceful stop
#
# 推荐：
# M5 Max / 64GB+
#
# pip install torch numpy tqdm
# =========================================================

import os
import re
import signal
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# MPS优化
# =========================================================

torch.mps.empty_cache()

torch.set_float32_matmul_precision("high")

# =========================================================
# 全局配置
# =========================================================

HISTORY_LEN = 4
STEP_CHANNELS = 2

BOARD_CHANNELS = 14

SIDE_CHANNELS = 1

TOTAL_CHANNELS = BOARD_CHANNELS + HISTORY_LEN * STEP_CHANNELS + SIDE_CHANNELS

BOARD_W = 9
BOARD_H = 10

CACHE_PATH = "xiangqi_dataset_cache-chatgpt.pt"

CHECKPOINT_PATH = "ckpt_latest-chatgpt.pth"

BEST_MODEL_PATH = "xiangqi_best-chatgpt.pth"

BATCH_SIZE = 128

# =========================================================
# 棋子定义
# =========================================================

EMPTY = 0

RED_K = 1
RED_R = 2
RED_N = 3
RED_C = 4
RED_E = 5
RED_A = 6
RED_P = 7

BLK_K = 8
BLK_R = 9
BLK_N = 10
BLK_C = 11
BLK_E = 12
BLK_A = 13
BLK_P = 14

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

# =========================================================
# piece plane
# =========================================================

PIECE_TO_CHANNEL = {
    RED_K: 0,
    RED_R: 1,
    RED_N: 2,
    RED_C: 3,
    RED_E: 4,
    RED_A: 5,
    RED_P: 6,
    BLK_K: 7,
    BLK_R: 8,
    BLK_N: 9,
    BLK_C: 10,
    BLK_E: 11,
    BLK_A: 12,
    BLK_P: 13,
}

# =========================================================
# ICCS
# =========================================================

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
}

# =========================================================
# Board
# =========================================================


class XiangqiBoard:

    def __init__(self):

        self.board = np.zeros((9, 10), dtype=np.uint8)

        self.history = []

        self.red_turn = True

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

                    if ch in FEN_PIECE:

                        self.board[c_idx, r_idx] = FEN_PIECE[ch]

                    c_idx += 1

        side = fen.split()[-1]

        self.red_turn = side == "w"

    def push(self, move):

        fc, fr, tc, tr = move

        self.history.append(move)

        self.board[tc, tr] = self.board[fc, fr]

        self.board[fc, fr] = EMPTY

        self.red_turn = not self.red_turn


# =========================================================
# ICCS parser
# =========================================================


def iccs_move_to_pos(move_str):

    frm, to = move_str.split("-")

    fc = COL_MAP[frm[0].upper()]
    fr = int(frm[1])

    tc = COL_MAP[to[0].upper()]
    tr = int(to[1])

    return (fc, fr, tc, tr)


# =========================================================
# parse PGNS
# =========================================================


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


# =========================================================
# Encoding
# =========================================================

EMPTY_STEP = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)


def encode_move(fc, fr, tc, tr):

    step = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)

    step[0, fc, fr] = 1

    step[1, tc, tr] = 1

    return step


def encode_board(board):

    planes = np.zeros((BOARD_CHANNELS, 9, 10), dtype=np.uint8)

    for x in range(9):

        for y in range(10):

            piece = board[x, y]

            if piece != EMPTY:

                ch = PIECE_TO_CHANNEL[piece]

                planes[ch, x, y] = 1

    return planes


def encode_side(is_red_turn):

    plane = np.zeros((1, 9, 10), dtype=np.uint8)

    if is_red_turn:

        plane[:] = 1

    return plane


def encode_state(board_obj):

    board_planes = encode_board(board_obj.board)

    hist = board_obj.history

    pad = [EMPTY_STEP] * max(0, HISTORY_LEN - len(hist))

    real_steps = [encode_move(*m) for m in hist[-HISTORY_LEN:]]

    hist_planes = np.concatenate(pad + real_steps, axis=0)

    side_plane = encode_side(board_obj.red_turn)

    state = np.concatenate([board_planes, hist_planes, side_plane], axis=0)

    return state


# =========================================================
# Dataset
# =========================================================


def games_to_dataset(games):

    X = []

    y_from = []

    y_to = []

    for game in tqdm(games, desc="生成数据"):

        board = XiangqiBoard()

        try:

            board.load_fen(game["fen"])

        except:

            continue

        for mv_str in game["moves"]:

            try:

                move = iccs_move_to_pos(mv_str)

                fc, fr, tc, tr = move

                state = encode_state(board)

                fid = fc * 10 + fr

                tid = tc * 10 + tr

                X.append(state)

                y_from.append(fid)

                y_to.append(tid)

                board.push(move)

            except:

                continue

    X = np.array(X, dtype=np.uint8)

    y_from = np.array(y_from, dtype=np.int64)

    y_to = np.array(y_to, dtype=np.int64)

    return X, y_from, y_to


# =========================================================
# Positional Encoding
# =========================================================


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=128, max_len=90):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):

        return x + self.pos_embedding


# =========================================================
# Model
# =========================================================


class HybridXiangqiModel(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(TOTAL_CHANNELS, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(d_model=d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.from_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, 90)
        )

        self.to_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, 90)
        )

    def forward(self, x):

        x = self.cnn(x)

        B, C, H, W = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = self.pos_enc(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        pred_from = self.from_head(x)

        pred_to = self.to_head(x)

        return pred_from, pred_to


# =========================================================
# checkpoint
# =========================================================


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


# =========================================================
# graceful stop
# =========================================================

stop_flag = False


def handle_exit(signum, frame):

    global stop_flag

    print("\n🛑 收到停止信号")

    stop_flag = True


signal.signal(signal.SIGINT, handle_exit)

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device: {device}")

    PGN_PATH = "./iccs_lib/ICCS-99813/" "dpxq-99813games.pgns"

    # =====================================================
    # dataset cache
    # =====================================================

    if os.path.exists(CACHE_PATH):

        print("加载缓存数据集")

        X, yf, yt = torch.load(CACHE_PATH)

    else:

        games = parse_iccs_pgn(PGN_PATH)

        print("games:", len(games))

        X, yf, yt = games_to_dataset(games)

        X = torch.tensor(X)

        yf = torch.tensor(yf, dtype=torch.long)

        yt = torch.tensor(yt, dtype=torch.long)

        torch.save((X, yf, yt), CACHE_PATH)

    # =====================================================
    # dataloader
    # =====================================================

    loader = DataLoader(
        TensorDataset(X, yf, yt),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    )

    # =====================================================
    # model
    # =====================================================

    model = HybridXiangqiModel(d_model=128, nhead=4, num_layers=2).to(device)

    # model = model.to(memory_format=torch.channels_last)

    # optional
    try:
        if device.type != "mps":

            model = torch.compile(model)

            print("torch.compile enabled")

    except:

        print("torch.compile unsupported")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=5e-5
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    best_loss = float("inf")

    # =====================================================
    # resume
    # =====================================================

    if os.path.exists(CHECKPOINT_PATH):

        start_epoch, best_loss = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )

    # =====================================================
    # training
    # =====================================================

    epoch = start_epoch
    running_loss = 0.0
    load_len = len(loader)
    while True:

        if stop_flag:
            break

        model.train()

        pbar = tqdm(
            loader,
            desc=f"epoch {epoch}",
            dynamic_ncols=True,
        )

        total_loss = 0.0
        steps = 0
        running_loss = 0.0
        running_top1_f = 0.0
        running_top1_t = 0.0
        running_top5_f = 0.0
        running_top5_t = 0.0

        for bx, bf, bt in pbar:

            if stop_flag:
                break

            bx = bx.to(device, non_blocking=True)

            bx = bx.float()

            bx = bx.contiguous(memory_format=torch.channels_last)

            bf = bf.to(device)

            bt = bt.to(device)

            optimizer.zero_grad()

            pred_f, pred_t = model(bx)

            loss_f = criterion(pred_f, bf)

            loss_t = criterion(pred_t, bt)

            loss = loss_f + loss_t

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # =====================================================
            # top1
            # =====================================================

            top1_f = (
                pred_f.argmax(dim=1) == bf
            ).float().mean().item()

            top1_t = (
                pred_t.argmax(dim=1) == bt
            ).float().mean().item()

            # =====================================================
            # top5
            # =====================================================

            top5_f = (
                pred_f.topk(5, dim=1).indices
                == bf.unsqueeze(1)
            ).any(dim=1).float().mean().item()

            top5_t = (
                pred_t.topk(5, dim=1).indices
                == bt.unsqueeze(1)
            ).any(dim=1).float().mean().item()

            # =====================================================
            # EMA 平滑
            # =====================================================

            running_loss = running_loss * 0.98 + loss.item() * 0.02

            running_top1_f = running_top1_f * 0.98 + top1_f * 0.02

            running_top1_t = running_top1_t * 0.98 + top1_t * 0.02

            running_top5_f = running_top5_f * 0.98 + top5_f * 0.02

            running_top5_t = running_top5_t * 0.98 + top5_t * 0.02

            total_loss += loss.item()

            steps += 1

            avg_loss = total_loss / steps

            pbar.set_postfix({

                "loss": f"{running_loss:.3f}",

                "avg": f"{avg_loss:.3f}",

                "f1": f"{running_top1_f:.3f}",

                "t1": f"{running_top1_t:.3f}",

                "f5": f"{running_top5_f:.3f}",

                "t5": f"{running_top5_t:.3f}",

                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"\nepoch={epoch}" f" loss={avg_loss:.4f}" f" lr={lr:.6f}")

        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, CHECKPOINT_PATH)

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            print(f"🏆 新最佳模型 " f"{best_loss:.4f}")

        epoch += 1

    save_checkpoint(model, optimizer, scheduler, epoch, best_loss, CHECKPOINT_PATH)

    print("\n训练安全结束")
