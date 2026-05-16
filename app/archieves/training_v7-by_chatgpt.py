import re
import numpy as np
from tqdm import tqdm

HISTORY_LEN = 4
STEP_CHANNELS = 2
BOARD_CHANNELS = 14
SIDE_CHANNELS = 1

TOTAL_CHANNELS = BOARD_CHANNELS + HISTORY_LEN * STEP_CHANNELS + SIDE_CHANNELS

COL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8}

EMPTY = 0

RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P = 1, 2, 3, 4, 5, 6, 7
BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P = 8, 9, 10, 11, 12, 13, 14

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


class XiangqiBoard:

    def __init__(self):

        self.board = np.zeros((9, 10), dtype=np.uint8)

        self.state = np.zeros((TOTAL_CHANNELS, 9, 10), dtype=np.uint8)

        self.red_turn = True

    def load_fen(self, fen):

        self.board.fill(0)

        self.state.fill(0)

        rows = fen.split()[0].split("/")

        for r, row in enumerate(rows):

            c = 0

            for ch in row:

                if ch.isdigit():

                    c += int(ch)

                else:

                    piece = FEN_PIECE[ch]

                    self.board[c, r] = piece

                    chn = PIECE_TO_CHANNEL[piece]

                    self.state[chn, c, r] = 1

                    c += 1

        self.red_turn = fen.split()[-1] == "w"

        if self.red_turn:

            self.state[-1, :, :] = 1

    def push(self, move):

        fc, fr, tc, tr = move

        moving = int(self.board[fc, fr])

        captured = int(self.board[tc, tr])

        moving_ch = PIECE_TO_CHANNEL[moving]

        self.board[fc, fr] = 0

        self.board[tc, tr] = moving

        self.state[moving_ch, fc, fr] = 0

        self.state[moving_ch, tc, tr] = 1

        if captured != EMPTY:

            captured_ch = PIECE_TO_CHANNEL[captured]

            self.state[captured_ch, tc, tr] = 0

        hist_start = BOARD_CHANNELS

        self.state[hist_start : hist_start + 6] = self.state[
            hist_start + 2 : hist_start + 8
        ].copy()

        self.state[hist_start + 6 : hist_start + 8] = 0

        self.state[hist_start + 6, fc, fr] = 1

        self.state[hist_start + 7, tc, tr] = 1

        self.red_turn = not self.red_turn

        self.state[-1, :, :] = 1 if self.red_turn else 0


def iccs_move_to_pos(move):

    if "-" not in move:

        return None

    frm, to = move.split("-")

    if len(frm) != 2 or len(to) != 2:
        return None

    if frm.isdigit():

        fc = int(frm[0])
        fr = int(frm[1])

        tc = int(to[0])
        tr = int(to[1])

    else:

        fc = COL_MAP[frm[0].upper()]
        fr = int(frm[1])

        tc = COL_MAP[to[0].upper()]
        tr = int(to[1])

    if not (0 <= fc < 9 and 0 <= tc < 9 and 0 <= fr < 10 and 0 <= tr < 10):
        return None

    return fc, fr, tc, tr


def parse_iccs_pgn(path):

    with open(path, "r", encoding="utf-8") as f:

        content = f.read()

    blocks = re.split(r"\n\n(?=\[Game)", content)

    games = []

    for block in blocks:

        if "[FEN" not in block:
            continue

        fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', block)

        if not fen_match:
            continue

        fen = fen_match.group(1)

        move_part = re.sub(r"\[.*?\]", "", block)

        move_part = re.sub(r"\d+\.", "", move_part)

        moves = move_part.split()

        games.append({"fen": fen, "moves": moves})

    return games


games = parse_iccs_pgn("./iccs_lib/ICCS-99813/dpxq-99813games.pgns")

X = []
yf = []
yt = []

for game in tqdm(games):

    board = XiangqiBoard()

    try:

        board.load_fen(game["fen"])

    except:
        continue

    for mv in game["moves"]:

        move = iccs_move_to_pos(mv)

        if move is None:
            continue

        fc, fr, tc, tr = move

        X.append(board.state.copy())

        yf.append(fc * 10 + fr)

        yt.append(tc * 10 + tr)

        board.push(move)

X = np.array(X, dtype=np.uint8)

yf = np.array(yf, dtype=np.uint8)

yt = np.array(yt, dtype=np.uint8)

print(X.shape)

np.save("X.npy", X)
np.save("yf.npy", yf)
np.save("yt.npy", yt)
