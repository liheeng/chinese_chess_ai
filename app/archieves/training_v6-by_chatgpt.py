import re
import os
import signal
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

torch.mps.empty_cache()
torch.set_float32_matmul_precision("high")

BOARD_W = 9
BOARD_H = 10

HISTORY_LEN = 4
STEP_CHANNELS = 2
BOARD_CHANNELS = 14
SIDE_CHANNELS = 1

TOTAL_CHANNELS = BOARD_CHANNELS + HISTORY_LEN * STEP_CHANNELS + SIDE_CHANNELS
TOTAL_SAMPLES = 8110000
BATCH_SIZE = 256
CNN_D_Model = 128
Transformer_N_Head = 8
Transformer_Num_Layers = 3

CHECKPOINT_PATH = "ckpt_latest-chatgpt.pth"
BEST_MODEL_PATH = "xiangqi_best-chatgpt.pth"

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
    "H": RED_N,

    "C": RED_C,

    "E": RED_E,
    "B": RED_E,

    "A": RED_A,
    "P": RED_P,

    "k": BLK_K,
    "r": BLK_R,
    "n": BLK_N,
    "h": BLK_N,

    "c": BLK_C,

    "e": BLK_E,
    "b": BLK_E,

    "a": BLK_A,
    "p": BLK_P,
}

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

PIECE_PLANES = np.eye(BOARD_CHANNELS, dtype=np.uint8)

EMPTY_STEP = np.zeros((STEP_CHANNELS, 9, 10), dtype=np.uint8)


# =========================================================
# 增量更新版 XiangqiBoard
# 不再每步重新 encode 23 channels
# 而是直接维护 state tensor
# =========================================================


class XiangqiBoard:

    def __init__(self):

        self.board = np.zeros((9, 10), dtype=np.uint8)

        self.history = []

        self.red_turn = True

        # 直接维护完整 state
        self.state = np.zeros((TOTAL_CHANNELS, 9, 10), dtype=np.uint8)

    # =====================================================
    # 初始化FEN
    # =====================================================

    def load_fen(self, fen):

        self.board.fill(EMPTY)

        self.state.fill(0)

        fen_board = fen.split()[0]

        rows = fen_board.split("/")

        for r_idx, row in enumerate(rows):

            c_idx = 0

            for ch in row:

                if ch.isdigit():

                    c_idx += int(ch)

                else:

                    piece = FEN_PIECE[ch]

                    self.board[c_idx, r_idx] = piece

                    channel = PIECE_TO_CHANNEL[int(piece)]

                    self.state[channel, c_idx, r_idx] = 1

                    c_idx += 1

        side = fen.split()[-1]

        self.red_turn = side == "w"

        # side plane
        if self.red_turn:

            self.state[-1, :, :] = 1

    # =====================================================
    # 获取当前state
    # =====================================================

    def get_state(self):

        return self.state

    # =====================================================
    # push move
    # 增量更新
    # =====================================================

    def push(self, move):

        fc, fr, tc, tr = move

        moving_piece = int(self.board[fc, fr])

        captured_piece = int(self.board[tc, tr])

        moving_channel = PIECE_TO_CHANNEL[moving_piece]

        # -------------------------------------------------
        # 更新board
        # -------------------------------------------------

        self.board[tc, tr] = moving_piece

        self.board[fc, fr] = EMPTY

        # -------------------------------------------------
        # 更新piece plane
        # -------------------------------------------------

        # 起点清空
        self.state[moving_channel, fc, fr] = 0

        # 终点落子
        self.state[moving_channel, tc, tr] = 1

        # 如果有吃子
        if captured_piece != EMPTY:

            captured_channel = PIECE_TO_CHANNEL[captured_piece]

            self.state[captured_channel, tc, tr] = 0

        # -------------------------------------------------
        # 更新history planes
        # 14~21
        # =================================================

        hist_start = BOARD_CHANNELS

        # 整体左移
        self.state[hist_start : hist_start + 6] = self.state[
            hist_start + 2 : hist_start + 8
        ]

        # 最后两个channel清零
        self.state[hist_start + 6 : hist_start + 8] = 0

        # 写入最新move
        self.state[hist_start + 6, fc, fr] = 1

        self.state[hist_start + 7, tc, tr] = 1

        # -------------------------------------------------
        # 更新side plane
        # -------------------------------------------------

        self.red_turn = not self.red_turn

        if self.red_turn:

            self.state[-1, :, :] = 1

        else:

            self.state[-1, :, :] = 0


def iccs_move_to_pos(move_str):

    move_str = move_str.strip()

    if "-" not in move_str:

        return None

    frm, to = move_str.split("-")

    # =====================================
    # 数字ICCS
    # 12-34
    # =====================================

    if (
        len(frm) == 2
        and len(to) == 2
        and frm.isdigit()
        and to.isdigit()
    ):

        fc = int(frm[0])

        fr = int(frm[1])

        tc = int(to[0])

        tr = int(to[1])

        return fc, fr, tc, tr

    # =====================================
    # 字母ICCS
    # A0-A1
    # =====================================

    if (
        len(frm) == 2
        and len(to) == 2
        and frm[0].isalpha()
    ):

        fc = COL_MAP[frm[0].upper()]

        fr = int(frm[1])

        tc = COL_MAP[to[0].upper()]

        tr = int(to[1])

        return fc, fr, tc, tr

    return None


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


class XiangqiIterableDataset(IterableDataset):
    def __init__(self, pgn_path):
        super().__init__()
        self.pgn_path = pgn_path

    def parse_games(self):
        with open(self.pgn_path, "r", encoding="utf-8") as f:
            content = f.read()

        game_blocks = re.split(r"\n\n(?=\[Game)", content)

        for block in game_blocks:

            if "[FEN" not in block:
                continue

            fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', block)

            if not fen_match:
                continue

            fen = fen_match.group(1)

            move_part = re.sub(r"\[.*?\]", "", block)

            move_part = re.sub(r"\d+\.", "", move_part)

            move_part = re.sub(r"\s+", " ", move_part).strip()

            move_tokens = [m for m in move_part.split() if "-" in m]

            yield {"fen": fen, "moves": move_tokens}

    def __iter__(self):
        for game in self.parse_games():

            board = XiangqiBoard()

            try:
                board.load_fen(game["fen"])
            except Exception as e:
                print(e)
                continue

            for mv_str in game["moves"]:

                try:
                    move = iccs_move_to_pos(mv_str)
                    if move is None:

                        continue
                    fc, fr, tc, tr = move

                    state = board.get_state().copy()

                    fid = fc * 10 + fr
                    tid = tc * 10 + tr

                    board.push(move)

                    yield (state, fid, tid)
                except Exception as e:
                    print(e)
                    continue


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=90):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding


class HybridXiangqiModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(TOTAL_CHANNELS, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(d_model)

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

    print(f"✅ 从 epoch {start_epoch} 继续训练")

    return start_epoch


stop_flag = False


def handle_exit(signum, frame):
    global stop_flag
    print("\n🛑 收到停止信号")
    stop_flag = True


signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device: {device}")

    PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"

    dataset = XiangqiIterableDataset(PGN_PATH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    model = HybridXiangqiModel(
        d_model=CNN_D_Model, nhead=Transformer_N_Head, num_layers=Transformer_Num_Layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float("inf")

    if os.path.exists(CHECKPOINT_PATH):

        start_epoch = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )

    epoch = start_epoch

    running_loss = 0.0
    while True:

        if stop_flag:
            break

        model.train()

        total_loss = 0.0
        steps = 0

        pbar = tqdm(
            loader,
            total=TOTAL_SAMPLES // BATCH_SIZE,
            desc=f"epoch {epoch}",
            dynamic_ncols=True,
        )

        for bx, bf, bt in pbar:

            if stop_flag:
                break

            bx = bx.to(device=device, dtype=torch.float32)
            bf = bf.to(device)
            bt = bt.to(device)

            bx = bx.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad()

            pred_f, pred_t = model(bx)

            loss = criterion(pred_f, bf) + criterion(pred_t, bt)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            steps += 1

            running_loss = running_loss * 0.98 + loss.item() * 0.02

            pbar.set_postfix(
                {
                    "loss": f"{running_loss:.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        scheduler.step()

        avg_loss = total_loss / steps

        lr = optimizer.param_groups[0]["lr"]

        print(f"\nepoch={epoch}" f" loss={avg_loss:.4f}" f" lr={lr:.6f}")

        save_checkpoint(model, optimizer, scheduler, epoch, CHECKPOINT_PATH)

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            print(f"🏆 新最佳模型 " f"{best_loss:.4f}")

        epoch += 1

    save_checkpoint(model, optimizer, scheduler, epoch, CHECKPOINT_PATH)

    print("\n训练安全结束")
