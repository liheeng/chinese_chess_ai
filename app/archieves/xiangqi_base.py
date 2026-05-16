import numpy as np
import re
from typing import Optional



class XiangQiBase():
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

    @staticmethod
    def in_border(c, r):
        return 0 <= c < 9 and 0 <= r < 10

    @staticmethod
    def in_red_palace(c, r):
        return 3 <= c <= 5 and 0 <= r <= 2

    @staticmethod
    def in_blk_palace(c, r):
        return 3 <= c <= 5 and 7 <= r <= 9

    @staticmethod
    def is_red(pid):
        return 1 <= pid <= 7

    @staticmethod
    def is_black(pid):
        return 8 <= pid <= 14

    @staticmethod
    def get_legal_moves(board, c0, r0):
        EMPTY = XiangQiBase.EMPTY
        RED_K, BLK_K = XiangQiBase.RED_K, XiangQiBase.BLK_K
        RED_A, BLK_A = XiangQiBase.RED_A, XiangQiBase.BLK_A
        RED_E, BLK_E = XiangQiBase.RED_E, XiangQiBase.BLK_E
        RED_N, BLK_N = XiangQiBase.RED_N, XiangQiBase.BLK_N
        RED_R, BLK_R = XiangQiBase.RED_R, XiangQiBase.BLK_R
        RED_C, BLK_C = XiangQiBase.RED_C, XiangQiBase.BLK_C
        RED_P, BLK_P = XiangQiBase.RED_P, XiangQiBase.BLK_P
        RED_PIECES = XiangQiBase.RED_PIECES
        BLK_PIECES = XiangQiBase.BLK_PIECES

        pid = board[c0, r0]
        if pid == EMPTY:
            return []
        moves = []

        # 将/帅
        if pid in (RED_K, BLK_K):
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dc, dr in dirs:
                c1, r1 = c0 + dc, r0 + dr
                if not XiangQiBase.in_border(c1, r1):
                    continue
                if XiangQiBase.is_red(pid) and not XiangQiBase.in_red_palace(c1, r1):
                    continue
                if XiangQiBase.is_black(pid) and not XiangQiBase.in_blk_palace(c1, r1):
                    continue
                if XiangQiBase.is_red(pid) and board[c1, r1] in RED_PIECES:
                    continue
                if XiangQiBase.is_black(pid) and board[c1, r1] in BLK_PIECES:
                    continue
                moves.append((c1, r1))

        # 士/仕
        elif pid in (RED_A, BLK_A):
            dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dc, dr in dirs:
                c1, r1 = c0 + dc, r0 + dr
                if not XiangQiBase.in_border(c1, r1):
                    continue
                if XiangQiBase.is_red(pid) and not XiangQiBase.in_red_palace(c1, r1):
                    continue
                if XiangQiBase.is_black(pid) and not XiangQiBase.in_blk_palace(c1, r1):
                    continue
                if XiangQiBase.is_red(pid) and board[c1, r1] in RED_PIECES:
                    continue
                if XiangQiBase.is_black(pid) and board[c1, r1] in BLK_PIECES:
                    continue
                moves.append((c1, r1))

        # 象/相
        elif pid in (RED_E, BLK_E):
            jumps = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
            eyes = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for idx, (dc, dr) in enumerate(jumps):
                ec, er = c0 + eyes[idx][0], r0 + eyes[idx][1]
                c1, r1 = c0 + dc, r0 + dr
                if not XiangQiBase.in_border(c1, r1) or not XiangQiBase.in_border(ec, er):
                    continue
                if board[ec, er] != EMPTY:
                    continue
                if XiangQiBase.is_red(pid) and r1 >= 5:
                    continue
                if XiangQiBase.is_black(pid) and r1 <= 4:
                    continue
                if XiangQiBase.is_red(pid) and board[c1, r1] in RED_PIECES:
                    continue
                if XiangQiBase.is_black(pid) and board[c1, r1] in BLK_PIECES:
                    continue
                moves.append((c1, r1))

        # 马
        elif pid in (RED_N, BLK_N):
            jumps = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            block = [(1, 0), (1, 0), (-1, 0), (-1, 0), (0, 1), (0, -1), (0, 1), (0, -1)]
            for idx, (dc, dr) in enumerate(jumps):
                bc, br = c0 + block[idx][0], r0 + block[idx][1]
                c1, r1 = c0 + dc, r0 + dr
                if not XiangQiBase.in_border(c1, r1):
                    continue
                if board[bc, br] != EMPTY:
                    continue
                if XiangQiBase.is_red(pid) and board[c1, r1] in RED_PIECES:
                    continue
                if XiangQiBase.is_black(pid) and board[c1, r1] in BLK_PIECES:
                    continue
                moves.append((c1, r1))

        # 车
        elif pid in (RED_R, BLK_R):
            for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                c1, r1 = c0 + dc, r0 + dr
                while XiangQiBase.in_border(c1, r1):
                    if board[c1, r1] != EMPTY:
                        if (XiangQiBase.is_red(pid) and board[c1, r1] not in RED_PIECES) or (
                            XiangQiBase.is_black(pid) and board[c1, r1] not in BLK_PIECES
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
                while XiangQiBase.in_border(c1, r1):
                    if not jump:
                        if board[c1, r1] == EMPTY:
                            moves.append((c1, r1))
                        else:
                            jump = True
                    else:
                        if board[c1, r1] != EMPTY:
                            if (XiangQiBase.is_red(pid) and board[c1, r1] not in RED_PIECES) or (
                                XiangQiBase.is_black(pid) and board[c1, r1] not in BLK_PIECES
                            ):
                                moves.append((c1, r1))
                            break
                    c1 += dc
                    r1 += dr

        # 兵/卒
        elif pid in (RED_P, BLK_P):
            if XiangQiBase.is_red(pid):
                dirs = [(0, -1)]
                if r0 <= 4:
                    dirs += [(1, 0), (-1, 0)]
            else:
                dirs = [(0, 1)]
                if r0 >= 5:
                    dirs += [(1, 0), (-1, 0)]
            for dc, dr in dirs:
                c1, r1 = c0 + dc, r0 + dr
                if not XiangQiBase.in_border(c1, r1):
                    continue
                if XiangQiBase.is_red(pid) and board[c1, r1] in RED_PIECES:
                    continue
                if XiangQiBase.is_black(pid) and board[c1, r1] in BLK_PIECES:
                    continue
                moves.append((c1, r1))
        return moves

    @staticmethod
    def get_legal_mask(board, turn):
        from_mask = np.full(90, -np.inf, dtype=np.float32)
        to_mask = np.full(90, -np.inf, dtype=np.float32)
        allow_pieces = XiangQiBase.RED_PIECES if turn == 0 else XiangQiBase.BLK_PIECES

        for c in range(9):
            for r in range(10):
                pid = board[c, r]
                if pid not in allow_pieces:
                    continue
                fid = c * 10 + r
                from_mask[fid] = 0.0
                for c1, r1 in XiangQiBase.get_legal_moves(board, c, r):
                    tid = c1 * 10 + r1
                    to_mask[tid] = 0.0
        return from_mask, to_mask

    @staticmethod
    def board_to_piece_planes(board_arr):
        """
        输入 (9,10) 整数棋盘
        输出 (14,9,10) 二值平面，每一类棋子单独一层
        """
        planes = np.zeros((14, 9, 10), dtype=np.float32)
        for idx, pid in enumerate(XiangQiBase.ALL_PIECE_IDS):
            mask = (board_arr == pid).astype(np.float32)
            planes[idx] = mask
        return planes

    @staticmethod
    def get_side_planes(turn: int):
        """
        4层：红回合、黑回合、九宫掩码、河界掩码
        turn:0红 1黑
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

        # 九宫
        palace = np.zeros((1, 9, 10), dtype=np.float32)
        palace[0, 3:6, 0:3] = 1.0
        palace[0, 3:6, 7:10] = 1.0

        # 河界分区
        river = np.ones((1, 9, 10), dtype=np.float32)

        return np.concatenate([red_turn, blk_turn, palace, river], axis=0)

    @staticmethod
    def iccs_move_to_pos(move_str):
        """将ICCS走法字符串转为坐标 (fc, fr, tc, tr)"""
        if "-" not in move_str:
            return None
        parts = move_str.strip().split("-")
        if len(parts) != 2:
            return None
        frm, to = parts
        if len(frm) < 2 or len(to) < 2:
            return None
        if frm[0].isdigit():
            fc = int(frm[0])
            fr = int(frm[1])
            tc = int(to[0])
            tr = int(to[1])
        else:
            fc = XiangQiBase.COL_MAP[frm[0].upper()]
            fr = int(frm[1])
            tc = XiangQiBase.COL_MAP[to[0].upper()]
            tr = int(to[1])
        if not (0 <= fc < 9 and 0 <= tc < 9 and 0 <= fr < 10 and 0 <= tr < 10):
            return None
        return fc, fr, tc, tr

    @staticmethod
    def encode_move(fc, fr, tc, tr):
        """编码一步棋为 2 层轨迹平面"""
        step = np.zeros((2, 9, 10), dtype=np.float32)
        step[0, fc, fr] = 1.0
        step[1, tc, tr] = 1.0
        return step

    @staticmethod
    def board_to_state(board_arr, history_moves, turn):
        """
        将棋盘 + 历史走法 + 回合编码为完整输入张量
        返回 (IN_CHANNELS, 9, 10)
        """
        HISTORY_LEN = XiangQiBase.HISTORY_LEN
        STEP_CHANNELS = XiangQiBase.STEP_CHANNELS

        # 历史轨迹
        pad = [np.zeros((STEP_CHANNELS, 9, 10), dtype=np.float32)] * max(
            0, HISTORY_LEN - len(history_moves)
        )
        real_steps = [XiangQiBase.encode_move(*m) for m in history_moves[-HISTORY_LEN:]]
        hist = np.concatenate(pad + real_steps, axis=0)

        # 棋子平面
        pieces = XiangQiBase.board_to_piece_planes(board_arr)

        # 规则平面
        side = XiangQiBase.get_side_planes(turn)

        return np.concatenate([hist, pieces, side], axis=0)


class XiangqiBoard:
    """完整的中国象棋棋盘类，支持 FEN 加载、走法生成、规则判定、状态编码"""

    def __init__(self):
        self.board = np.zeros((9, 10), dtype=int)
        self.history: list = []  # 存储走法 (fc, fr, tc, tr)
        self.snapshot_history: list = []  # 存储历史棋盘快照（用于悔棋）
        self.turn = 0  # 0=红方, 1=黑方

    # ───────────── 棋盘初始化和FEN ─────────────

    def load_fen(self, fen: str):
        """从 FEN 字符串加载棋盘"""
        self.board.fill(XiangQiBase.EMPTY)
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
                    if ch in XiangQiBase.FEN_PIECE and c_idx < 9:
                        self.board[c_idx, r_idx] = XiangQiBase.FEN_PIECE[ch]
                    c_idx += 1

    def generate_fen(self) -> str:
        """将当前棋盘导出为 FEN 字符串"""
        rows = []
        for r in range(10):
            row = ""
            empty_count = 0
            for c in range(9):
                pid = self.board[c, r]
                if pid == XiangQiBase.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    # 反向查找 FEN_PIECE
                    for fen_ch, piece_id in XiangQiBase.FEN_PIECE.items():
                        if piece_id == pid:
                            row += fen_ch
                            break
            if empty_count > 0:
                row += str(empty_count)
            rows.append(row)
        fen_board = "/".join(rows)
        turn_char = "w" if self.turn == 0 else "b"
        return f"{fen_board} {turn_char} - - 0 1"

    def reset(self):
        """重置棋盘到初始局面"""
        fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/RNBAKABNR w - - 0 1"
        self.load_fen(fen)

    # ───────────── 走法操作 ─────────────

    def push(self, move):
        """执行一步棋，move=(fc, fr, tc, tr)"""
        fc, fr, tc, tr = move
        self.snapshot_history.append(self.board.copy())
        self.history.append((fc, fr, tc, tr))
        self.board[tc, tr] = self.board[fc, fr]
        self.board[fc, fr] = XiangQiBase.EMPTY
        self.turn = 1 - self.turn

    def undo(self) -> bool:
        """悔棋一步，返回是否成功"""
        if not self.history:
            return False
        self.history.pop()
        self.board = self.snapshot_history.pop()
        self.turn = 1 - self.turn
        return True

    def copy(self) -> "XiangqiBoard":
        """深拷贝当前棋盘"""
        new_board = XiangqiBoard()
        new_board.board = self.board.copy()
        new_board.history = self.history.copy()
        new_board.snapshot_history = [s.copy() for s in self.snapshot_history]
        new_board.turn = self.turn
        return new_board

    def snapshot(self):
        """返回当前棋盘快照"""
        return self.board.copy()

    # ───────────── 走法合法性 ─────────────

    def get_legal_moves(self) -> list:
        """获取当前回合所有合法走法列表 [(fc,fr,tc,tr), ...]"""
        moves = []
        allow_pieces = (
            XiangQiBase.RED_PIECES if self.turn == 0 else XiangQiBase.BLK_PIECES
        )
        for c in range(9):
            for r in range(10):
                pid = self.board[c, r]
                if pid not in allow_pieces:
                    continue
                for tc, tr in XiangQiBase.get_legal_moves(self.board, c, r):
                    # 模拟走法，检查是否导致己方被将
                    captured = self.board[tc, tr]
                    self.board[tc, tr] = pid
                    self.board[c, r] = XiangQiBase.EMPTY
                    if not self.is_check():
                        moves.append((c, r, tc, tr))
                    # 还原
                    self.board[c, r] = pid
                    self.board[tc, tr] = captured
        return moves

    def is_legal_move(self, move) -> bool:
        """检查 move=(fc,fr,tc,tr) 是否为合法走法"""
        return move in self.get_legal_moves()

    # ───────────── 将军/将杀判定 ─────────────

    def find_king(self, is_red_side: bool) -> tuple:
        """查找指定方的将/帅位置，返回 (c, r)"""
        king_id = XiangQiBase.RED_K if is_red_side else XiangQiBase.BLK_K
        for c in range(9):
            for r in range(10):
                if self.board[c, r] == king_id:
                    return c, r
        return -1, -1

    def is_check(self) -> bool:
        """当前走棋方是否被将军"""
        # 检查对方棋子是否能吃掉己方将/帅
        enemy_pieces = (
            XiangQiBase.BLK_PIECES if self.turn == 0 else XiangQiBase.RED_PIECES
        )
        king_id = XiangQiBase.RED_K if self.turn == 0 else XiangQiBase.BLK_K

        # 找己方将/帅位置
        kc, kr = -1, -1
        for c in range(9):
            for r in range(10):
                if self.board[c, r] == king_id:
                    kc, kr = c, r
                    break
            if kc >= 0:
                break

        if kc < 0:
            return False

        # 检查对方所有棋子是否能攻击到将/帅
        for c in range(9):
            for r in range(10):
                pid = self.board[c, r]
                if pid not in enemy_pieces:
                    continue
                for tc, tr in XiangQiBase.get_legal_moves(self.board, c, r):
                    if tc == kc and tr == kr:
                        return True

        # 检查对面将/帅是否可以直接"对面"（将帅照面）
        opp_king_id = XiangQiBase.BLK_K if self.turn == 0 else XiangQiBase.RED_K
        okc, okr = -1, -1
        for c in range(9):
            for r in range(10):
                if self.board[c, r] == opp_king_id:
                    okc, okr = c, r
                    break
            if okc >= 0:
                break
        if kc == okc:
            # 同一列，中间无棋子
            min_r, max_r = min(kr, okr), max(kr, okr)
            blocked = False
            for r in range(min_r + 1, max_r):
                if self.board[kc, r] != XiangQiBase.EMPTY:
                    blocked = True
                    break
            if not blocked:
                return True

        return False

    def is_checkmate(self) -> bool:
        """是否被将杀（将死）"""
        if not self.is_check():
            return False
        return len(self.get_legal_moves()) == 0

    def is_stalemate(self) -> bool:
        """是否被困毙（无子可走但未被将）"""
        if self.is_check():
            return False
        return len(self.get_legal_moves()) == 0

    def is_game_over(self) -> bool:
        """游戏是否结束（将杀或困毙）"""
        return self.is_checkmate() or self.is_stalemate()

    # ───────────── 状态编码（用于神经网络输入） ─────────────

    def to_state(self) -> np.ndarray:
        """将当前棋盘编码为 (IN_CHANNELS, 9, 10) 张量"""
        return XiangQiBase.board_to_state(self.board, self.history, self.turn)

    def get_legal_mask(self):
        """获取当前回合的合法走法掩码"""
        from_mask = np.full(90, -np.inf, dtype=np.float32)
        to_mask = np.full(90, -np.inf, dtype=np.float32)
        for c, r, tc, tr in self.get_legal_moves():
            from_mask[c * 10 + r] = 0.0
            to_mask[tc * 10 + tr] = 0.0
        return from_mask, to_mask

    # ───────────── 显示 ─────────────

    def display(self):
        """打印当前棋盘"""
        piece_chars = {
            XiangQiBase.RED_K: "帅", XiangQiBase.RED_R: "车",
            XiangQiBase.RED_N: "马", XiangQiBase.RED_C: "炮",
            XiangQiBase.RED_E: "相", XiangQiBase.RED_A: "仕",
            XiangQiBase.RED_P: "兵",
            XiangQiBase.BLK_K: "将", XiangQiBase.BLK_R: "車",
            XiangQiBase.BLK_N: "馬", XiangQiBase.BLK_C: "砲",
            XiangQiBase.BLK_E: "象", XiangQiBase.BLK_A: "士",
            XiangQiBase.BLK_P: "卒",
        }
        print(f"  {' '.join('abcdefghi')}")
        for r in range(10):
            row_str = f"{9-r} "
            for c in range(9):
                pid = self.board[c, r]
                if pid == XiangQiBase.EMPTY:
                    row_str += ". "
                else:
                    row_str += piece_chars.get(pid, "?") + " "
            print(row_str)
        turn_str = "红方" if self.turn == 0 else "黑方"
        print(f"\n当前走棋方: {turn_str}")
        if self.is_check():
            print("⚠️  将军!")
        if self.is_checkmate():
            print("🏁  将杀!")
        if self.is_stalemate():
            print("🏁  困毙!")

    def __repr__(self):
        return f"XiangqiBoard(turn={'red' if self.turn==0 else 'black'}, moves={len(self.history)})"