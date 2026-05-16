import numpy as np
from archieves.deepseek.chinese_chess_ai.core import Board, BOARD_W, BOARD_H


class ChineseChessEncoder:
    """
    中国象棋局面编码器，输出 26 通道 (10,9)
    通道 0-13: 14个棋子平面
    通道 14-21: 4步历史（每步2个通道：移动棋子类型 + 移动方向编码）
    通道 22: 轮到红方走
    通道 23: 红方九宫区域
    通道 24: 黑方九宫区域
    通道 25: 河界
    """

    def __init__(self, history_len=4):
        self.history_len = history_len
        self.num_planes = 26
        # 棋子类型索引 (0-6)
        self.piece_type_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
        }  # 帅士象马车炮兵

    def encode_position(self, board: Board) -> np.ndarray:
        """编码当前棋盘，返回 (26,10,9) numpy 数组"""
        planes = np.zeros((self.num_planes, BOARD_H, BOARD_W), dtype=np.float32)

        # 0-13: 棋子平面
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                piece = board.board[y][x]
                if piece is None:
                    continue
                color = 0 if piece < 7 else 1  # 0红,1黑
                piece_type = piece % 7  # 0-6
                plane_idx = color * 7 + piece_type
                planes[plane_idx, y, x] = 1.0

        # 22: 轮到红方走
        planes[22, :, :] = 1.0 if board.turn == 0 else 0.0

        # 23: 红方九宫 (x 3-5, y 7-9)
        for y in range(7, 10):
            for x in range(3, 6):
                planes[23, y, x] = 1.0

        # 24: 黑方九宫 (x 3-5, y 0-2)
        for y in range(0, 3):
            for x in range(3, 6):
                planes[24, y, x] = 1.0

        # 25: 河界 (y=5 是河界中线)
        planes[25, 5, :] = 1.0

        return planes

    def encode_with_history(self, board_history):
        """
        board_history: 最近 self.history_len 个 Board 对象（最近的在前）
        返回 (26,10,9)
        """
        # 补齐历史
        while len(board_history) < self.history_len:
            board_history.insert(0, Board())
        board_history = board_history[-self.history_len :]

        # 当前局面通道
        current_planes = self.encode_position(board_history[-1])
        # 历史走法通道（14-21）
        history_planes = self._encode_history_moves(board_history)

        # 合并：当前棋子平面(0-13) + 历史走法(14-21) + 辅助(22-25)
        final_planes = np.concatenate(
            [current_planes[0:14], history_planes, current_planes[22:26]], axis=0
        )
        return final_planes  # shape (26,10,9)

    def _encode_history_moves(self, board_history):
        """
        编码历史走法：每步用2个通道
        通道1: 移动棋子的类型 (0-6)
        通道2: 移动方向编码 (0-7 表示8个方向)
        """
        hist_planes = np.zeros(
            (self.history_len * 2, BOARD_H, BOARD_W), dtype=np.float32
        )
        # 这里简化实现：只记录“是否有移动”，实际可以更丰富
        # 为了演示，将每个历史局面中发生过移动的格子标记为1
        for step in range(self.history_len - 1):
            # 模拟相邻两步的差异（实际需要存储移动记录，此处略）
            pass
        return hist_planes
