import re
from typing import List, Tuple
from torch.utils.data import Dataset
from archieves.deepseek.chinese_chess_ai.core import Board
from archieves.deepseek.chinese_chess_ai.board_encoder import ChineseChessEncoder


def parse_iccs_file(file_path: str) -> List[Tuple[List[str], str]]:
    """
    解析 ICCS 文本文件，返回棋谱列表
    每个棋谱: (move_strings列表, 结果)
    结果: "1-0" / "0-1" / "1/2-1/2"
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 # Game 分割
    raw_games = re.split(r"#\s*Game\s+\d+", content)
    raw_games = [g.strip() for g in raw_games if g.strip()]

    # 定义走法匹配模式（支持 ICCS 标准和 UCI 格式）
    move_pattern = re.compile(r"[A-Za-z]?\d+[a-z]?(?:-[A-Za-z]?\d+[a-z]?)?")

    parsed = []
    for game_text in raw_games:
        # 提取结果
        result_match = re.search(r"(1-0|0-1|1/2-1/2)\s*$", game_text)
        if not result_match:
            continue
        result = result_match.group(1)

        # 提取所有走法
        moves = []
        # 按行处理，去除行首序号
        for line in game_text.split("\n"):
            # 去掉行首序号（如 "1. " 或 "1、 "）
            line = re.sub(r"^\s*\d+[\.\、]\s*", "", line.strip())
            if not line:
                continue
            # 在行内查找所有匹配走法模式的子串
            found = move_pattern.findall(line)
            for token in found:
                # 过滤掉结果字符串（避免误判）
                if token in ("1-0", "0-1", "1/2-1/2"):
                    continue
                moves.append(token)
        if moves:
            parsed.append((moves, result))
    return parsed


class ICCSDataset(Dataset):
    def __init__(self, iccs_paths: List[str], max_games: int | None = None):
        self.encoder = ChineseChessEncoder(history_len=4)
        self.samples = []  # 每个元素: (state_planes, move_index, value)
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_map()

        for path in iccs_paths:
            games = parse_iccs_file(path)
            if max_games:
                games = games[:max_games]
            for moves, result in games:
                self._add_game(moves, result)

    def _build_move_map(self):
        """构建 90*90 的索引映射，实际只保留可能的走法，但为了简单，全量映射"""
        idx = 0
        for from_pos in range(90):
            for to_pos in range(90):
                self.move_to_idx[(from_pos, to_pos)] = idx
                self.idx_to_move[idx] = (from_pos, to_pos)
                idx += 1
        self.num_moves = idx  # 8100

    # iccs_loader.py 关键修改（简化历史记录）
    def _add_game(self, move_strings: List[str], result: str):
        board = Board()
        # 使用 FEN 快照列表，存储 FEN 字符串而不是 Board 对象
        history_fens = []
        final_value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)

        for move_str in move_strings:
            move_coord = self._iccs_to_coord(move_str, board)
            if move_coord is None:
                break
            from_pos, to_pos = move_coord
            # 保存当前局面样本
            if len(history_fens) > 0:
                # 从 FEN 重建棋盘并编码
                hist_board = Board()
                hist_board.set_fen(
                    history_fens[-1]
                )  # 需要实现 set_fen 方法，或直接用历史 board 快照
                # 简便做法：直接使用当前 board 的拷贝，但注意 board 会变化，所以需要在走法前拷贝
                # 我们在走法前保存当前 board 的深拷贝
            # 执行走法
            if board.is_legal_move(from_pos, to_pos):
                # 先保存当前局面（深拷贝）
                import copy

                hist_board = copy.deepcopy(board)
                history_fens.append(hist_board)  # 存储 Board 对象
                board.push(from_pos, to_pos)
                # 保持历史长度
                if len(history_fens) > 4:
                    history_fens.pop(0)
                # 生成样本（略）

    def _iccs_to_coord(self, move_str: str, board: Board):
        """将 ICCS 走法字符串转换为 (from_pos, to_pos)，坐标格式 (x,y)"""
        # 简化实现：支持 UCI 格式例如 "h2e2" 以及 ICCS 带棋子类型格式
        # 这里使用正则提取数字字母
        # 格式1: 字母数字-字母数字 -> C3e-4e
        m = re.match(r"([A-Z]?)([0-9][a-z]?)-([0-9][a-z]?)", move_str)
        if m:
            _, src, dst = m.groups()
            src_x, src_y = self._coord_to_xy(src)
            dst_x, dst_y = self._coord_to_xy(dst)
            return ((src_x, src_y), (dst_x, dst_y))
        # 格式2: 纯数字字母如 "h2e2"
        m2 = re.match(r"([a-i][0-9])([a-i][0-9])", move_str.lower())
        if m2:
            src, dst = m2.groups()
            src_x, src_y = self._coord_to_xy(src)
            dst_x, dst_y = self._coord_to_xy(dst)
            return ((src_x, src_y), (dst_x, dst_y))
        return None

    def _coord_to_xy(self, coord: str):
        """将 'e2' 转换为 (x,y)  其中 x: 0-8, y: 0-9, e2 表示纵线e(5), 横线2"""
        # 例如 "3e": 数字是纵线索引+1, 字母是横线( a=0, b=1,... i=8)
        # 但实际ICCS常见是 'e2' 表示 纵线 e(5), 横线2
        if len(coord) == 2:
            file_char = coord[0]  # a-i
            rank = int(coord[1])  # 0-9
            file_idx = ord(file_char) - ord("a")
            return (file_idx, rank)
        elif len(coord) == 3:
            # 如 "12a" 极少，忽略
            return (0, 0)
        return (0, 0)

    def _encode_state(self, history_boards):
        return self.encoder.encode_with_history(history_boards)
        # """编码历史棋盘为 (72,10,9) 张量"""
        # # 简化的编码: 每个棋盘编码为 14个棋子通道 + 4辅助 = 18通道，4步历史 = 72通道
        # # 这里返回 numpy 数组
        # planes = []
        # for b in history_boards[-4:]:
        #     # 每个棋盘编码 18 通道
        #     board_planes = np.zeros((18, 10, 9), dtype=np.float32)
        #     # 棋子通道：红7+黑7
        #     for y in range(10):
        #         for x in range(9):
        #             piece = b.board[y][x]
        #             if piece is not None:
        #                 color = 0 if piece < 7 else 1
        #                 piece_type = piece % 7
        #                 plane_idx = color * 7 + piece_type
        #                 board_planes[plane_idx, y, x] = 1.0
        #     # 辅助通道: 轮到谁 (0:红,1:黑) 作为第14通道
        #     board_planes[14, :, :] = 1.0 if b.turn == 0 else 0.0
        #     # 其他辅助通道可置0
        #     planes.append(board_planes)
        # # 补全到4步
        # while len(planes) < 4:
        #     planes.insert(0, np.zeros((18, 10, 9)))
        # return np.concatenate(planes, axis=0)  # (72,10,9)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, move_idx, value = self.samples[idx]
        return state, move_idx, value
