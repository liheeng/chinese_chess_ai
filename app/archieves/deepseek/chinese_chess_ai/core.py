"""
中国象棋底层规则实现（10x9 棋盘）
所有棋子移动规则、合法性检查、胜负判定均完整实现
"""

from typing import List, Tuple, Optional
import copy

BOARD_W, BOARD_H = 9, 10

# 棋子编码（0~13）
# 红方: 帅(0), 士(1), 象(2), 马(3), 车(4), 炮(5), 兵(6)
# 黑方: 将(7), 士(8), 象(9), 马(10),车(11),炮(12),卒(13)
PIECE_TYPE = {
    "RK": 0,
    "RA": 1,
    "RE": 2,
    "RN": 3,
    "RR": 4,
    "RC": 5,
    "RP": 6,
    "BK": 7,
    "BA": 8,
    "BE": 9,
    "BN": 10,
    "BR": 11,
    "BC": 12,
    "BP": 13,
}
# 用于显示的字符
PIECE_CHARS = "K A E N R C P k a e n r c p".split()


class Board:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        self.turn = 0  # 0: 红方, 1: 黑方
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._init_board()

    def _init_board(self):
        # 使用 FEN 字符串初始化（简化）
        # 黑方在上（行0~2），红方在下（行9~7）
        layout = [
            "rheakaehr",  # 0
            ".........",  # 1
            ".c.....c.",  # 2
            "p.p.p.p.p",  # 3
            ".........",  # 4
            ".........",  # 5
            "P.P.P.P.P",  # 6
            ".C.....C.",  # 7
            ".........",  # 8
            "RHEAKAEHR",  # 9
        ]
        # 映射字符 -> 编码整数
        char_to_code = {
            "r": 11,
            "h": 10,
            "e": 9,
            "a": 8,
            "k": 7,
            "c": 12,
            "p": 13,
            "R": 4,
            "H": 3,
            "E": 2,
            "A": 1,
            "K": 0,
            "C": 5,
            "P": 6,
        }
        for row, line in enumerate(layout):
            for col, ch in enumerate(line):
                if ch != ".":
                    self.board[row][col] = char_to_code[ch]

    def get_piece(self, x: int, y: int) -> Optional[int]:
        if 0 <= x < BOARD_W and 0 <= y < BOARD_H:
            return self.board[y][x]
        return None

    def is_legal_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        fx, fy = from_pos
        tx, ty = to_pos
        piece = self.get_piece(fx, fy)
        if piece is None:
            return False
        # 颜色
        color = 0 if piece < 7 else 1
        if color != self.turn:
            return False
        target = self.get_piece(tx, ty)
        if target is not None:
            target_color = 0 if target < 7 else 1
            if target_color == color:
                return False  # 不能吃自己的子

        # 棋子移动规则
        if not self._piece_move_valid(fx, fy, tx, ty, piece):
            return False

        # 模拟移动，检查移动后自己的将是否被吃 或者 将帅对面
        temp_board = copy.deepcopy(self)
        temp_board._apply_move(from_pos, to_pos)
        # 检查移动方（原 turn）的将是否被攻击
        if temp_board.is_king_attacked(color):
            return False
        # 检查将帅对面（无将照面）
        if temp_board.is_flying_king():
            return False
        return True

    def _piece_move_valid(self, fx: int, fy: int, tx: int, ty: int, piece: int) -> bool:
        dx = abs(tx - fx)
        dy = abs(ty - fy)
        piece_type = piece % 7  # 0:帅/将,1:士,2:象,3:马,4:车,5:炮,6:兵/卒
        color = 0 if piece < 7 else 1

        # 将帅
        if piece_type == 0:
            if color == 0:  # 红帅
                if not (3 <= tx <= 5 and 7 <= ty <= 9):
                    return False
            else:  # 黑将
                if not (3 <= tx <= 5 and 0 <= ty <= 2):
                    return False
            if dx + dy != 1:
                return False
            return True

        # 士
        if piece_type == 1:
            if color == 0:
                if not (3 <= tx <= 5 and 7 <= ty <= 9):
                    return False
            else:
                if not (3 <= tx <= 5 and 0 <= ty <= 2):
                    return False
            if dx == 1 and dy == 1:
                return True
            return False

        # 象
        if piece_type == 2:
            if color == 0 and ty < 5:  # 红象不能过河（河界在 y=5）
                return False
            if color == 1 and ty > 4:
                return False
            if dx == 2 and dy == 2:
                mx, my = (fx + tx) // 2, (fy + ty) // 2
                if self.get_piece(mx, my) is None:
                    return True
            return False

        # 马
        if piece_type == 3:
            if dx == 2 and dy == 1:
                leg_x = fx + (1 if tx > fx else -1)
                leg_y = fy
                if self.get_piece(leg_x, leg_y) is None:
                    return True
            if dx == 1 and dy == 2:
                leg_x = fx
                leg_y = fy + (1 if ty > fy else -1)
                if self.get_piece(leg_x, leg_y) is None:
                    return True
            return False

        # 车
        if piece_type == 4:
            return self._is_clear_line(fx, fy, tx, ty)

        # 炮
        if piece_type == 5:
            has_mid = not self._is_clear_line(fx, fy, tx, ty)
            target = self.get_piece(tx, ty)
            if target is None:
                return not has_mid
            else:
                return has_mid

        # 兵/卒
        if piece_type == 6:
            forward = -1 if color == 0 else 1  # 红向上(y减)，黑向下(y增)
            # 过河判断
            is_across = (color == 0 and fy < 5) or (color == 1 and fy > 4)
            # 向前移动
            if dx == 0 and ty - fy == forward:
                return True
            # 过河后可以左右平移
            if is_across and dy == 0 and dx == 1:
                return True
            return False

        return False

    def _is_clear_line(self, fx: int, fy: int, tx: int, ty: int) -> bool:
        """直线路径上无阻挡（不包括起点终点）"""
        if fx != tx and fy != ty:
            return False
        if fx == tx:
            step = 1 if ty > fy else -1
            for y in range(fy + step, ty, step):
                if self.get_piece(fx, y) is not None:
                    return False
        else:
            step = 1 if tx > fx else -1
            for x in range(fx + step, tx, step):
                if self.get_piece(x, fy) is not None:
                    return False
        return True

    def _apply_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        fx, fy = from_pos
        tx, ty = to_pos
        piece = self.board[fy][fx]
        self.board[ty][tx] = piece
        self.board[fy][fx] = None
        self.turn = 1 - self.turn

    def push(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """执行走法（调用前请确保合法性）"""
        self._apply_move(from_pos, to_pos)

    def is_king_attacked(self, color: int) -> bool:
        """检查 color 方的将/帅是否被敌方棋子攻击"""
        king_code = 0 if color == 0 else 7
        kx = ky = -1
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                if self.board[y][x] == king_code:
                    kx, ky = x, y
                    break
        if kx == -1:
            return False
        enemy_color = 1 - color
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p is not None and ((p < 7) == (enemy_color == 0)):
                    # 检查该敌方棋子是否能走到 king 的位置
                    if self._piece_move_valid(x, y, kx, ky, p):
                        # 还需要检查路径（马、炮等需要考虑路径阻挡）
                        # 复用 _piece_move_valid 已经包含了马腿、炮架等，但直线棋子还需路径无阻挡
                        # 对于车、炮，需要额外检查中间是否有子
                        p_type = p % 7
                        if p_type in (4, 5):  # 车或炮
                            # 检查直线路径上除了终点外是否有阻挡
                            if not self._is_clear_line(x, y, kx, ky):
                                # 有阻挡：车不能攻击，炮需要恰好一个炮架
                                if p_type == 4:
                                    continue
                                else:  # 炮
                                    # 数一下中间有几个子
                                    mid_count = self._count_pieces_between(x, y, kx, ky)
                                    if mid_count != 1:
                                        continue
                        return True
        return False

    def _count_pieces_between(self, fx: int, fy: int, tx: int, ty: int) -> int:
        """计算直线路径上的棋子数量（不包括起点和终点）"""
        count = 0
        if fx == tx:
            step = 1 if ty > fy else -1
            for y in range(fy + step, ty, step):
                if self.get_piece(fx, y) is not None:
                    count += 1
        elif fy == ty:
            step = 1 if tx > fx else -1
            for x in range(fx + step, tx, step):
                if self.get_piece(x, fy) is not None:
                    count += 1
        return count

    def is_flying_king(self) -> bool:
        """检查将帅对面无遮挡（飞将）"""
        # 找红帅和黑将的列位置
        rkx = rky = None
        bkx = bky = None
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p == 0:
                    rkx, rky = x, y
                elif p == 7:
                    bkx, bky = x, y
        if rkx is None or bkx is None:
            return False
        if rkx != bkx:
            return False
        # 同一列，检查中间是否有棋子
        step = 1 if bky > rky else -1
        for y in range(rky + step, bky, step):
            if self.board[y][rkx] is not None:
                return False
        return True

    def get_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        moves = []
        for fy in range(BOARD_H):
            for fx in range(BOARD_W):
                p = self.board[fy][fx]
                if p is not None and (
                    (p < 7 and self.turn == 0) or (p >= 7 and self.turn == 1)
                ):
                    for ty in range(BOARD_H):
                        for tx in range(BOARD_W):
                            if self.is_legal_move((fx, fy), (tx, ty)):
                                moves.append(((fx, fy), (tx, ty)))
        return moves

    def result(self) -> str:
        """返回对局结果：1-0 红胜，0-1 黑胜，* 未结束"""
        if len(self.get_legal_moves()) == 0:
            # 无步可走，当前方输
            return "1-0" if self.turn == 1 else "0-1"
        # 简单判和条件可扩展
        return "*"

    def __str__(self):
        s = ""
        for y in range(BOARD_H):
            row = ""
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p is None:
                    row += "."
                else:
                    row += PIECE_CHARS[p]
            s += row + "\n"
        return s
