import os
import torch
import pygame
import pygame.freetype  # 更好的字体引擎，支持中文

from doubao.cchess_doubao_deepseek_v1 import (
    # 全局配置
    IN_CHANNELS,
    HISTORY_LEN,
    STEP_CHANNELS,
    TRACE_END,
    PIECE_END,
    COL_MAP,
    EMPTY,
    # 棋子定义
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
    # 核心函数
    reconstruct_state,
    get_legal_mask,
    get_pseudo_legal_moves,
    iccs_move_to_pos,
    # 棋盘类
    XiangqiBoard,
    # 模型
    HybridXiangqiModel,
    PositionalEncoding,
)

# 文件路径
# ===================== 推理配置 =====================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 👇 模型路径和你的训练代码完全一致
# file_name = os.path.splitext(os.path.basename(__file__))[0]
BASE_DIR = "./data"
BEST_MODEL_PATH = f"{BASE_DIR}/xiangqi_best_cchess_doubao_deepseek_v1.pth"

# ===================== 图形化配置 =====================
CELL_SIZE = 60
BOARD_PADDING = 60  # 棋盘四周留白，避免棋子被裁剪
BOARD_WIDTH = 8 * CELL_SIZE  # 9个交叉点之间的总宽度
BOARD_HEIGHT = 9 * CELL_SIZE  # 10个交叉点之间的总高度
WINDOW_WIDTH = BOARD_WIDTH + BOARD_PADDING * 2 + 220
WINDOW_HEIGHT = BOARD_HEIGHT + BOARD_PADDING * 2
# 颜色
BG_COLOR = (245, 222, 179)
LINE_COLOR = (101, 67, 33)
RED_CHESS = (255, 0, 0)
BLACK_CHESS = (0, 0, 0)
WHITE = (255, 255, 255)
HIGHLIGHT_COLOR = (0, 255, 0)  # 仅新增高亮颜色，不改动原有配置

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("中国象棋AI对战")
clock = pygame.time.Clock()

# 鼠标交互全局变量 - 仅新增，不改动任何原有代码
selected_col = -1
selected_row = -1
legal_moves = []
mouse_x, mouse_y = 0, 0

# 游戏状态
GAME_OVER = False
WINNER_MSG = ""
GAME_OVER_COLOR = BLACK_CHESS

# ===================== 棋子中文映射表 =====================
CHESS_TEXT = {
    RED_K: "帅",
    RED_R: "车",
    RED_N: "马",
    RED_C: "炮",
    RED_E: "相",
    RED_A: "仕",
    RED_P: "兵",
    BLK_K: "将",
    BLK_R: "車",
    BLK_N: "馬",
    BLK_C: "砲",
    BLK_E: "象",
    BLK_A: "士",
    BLK_P: "卒",
}

# ===================== 新增：象棋规则兜底（仅修复将军bug，不碰AI逻辑） =====================
# 定义红黑棋子集合（用于判断将军）
RED_PIECES = {RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P}
BLK_PIECES = {BLK_K, BLK_R, BLK_N, BLK_C, BLK_E, BLK_A, BLK_P}


def is_in_check(board_arr, turn):
    """判断当前方是否被将军（原版规则，无任何魔改）"""
    king = RED_K if turn == 0 else BLK_K
    # 找老将位置
    kc, kr = -1, -1
    for c in range(9):
        for r in range(10):
            if board_arr[c, r] == king:
                kc, kr = c, r
                break
        if kc != -1:
            break
    # 检查对方棋子是否能吃老将
    enemy = BLK_PIECES if turn == 0 else RED_PIECES
    for c in range(9):
        for r in range(10):
            if board_arr[c, r] in enemy:
                moves = get_pseudo_legal_moves(board_arr, c, r)
                if (kc, kr) in moves:
                    return True
    return False


def get_all_legal_moves(board_arr, turn):
    """获取当前所有合法走法（原版规则）"""
    moves = []
    own = RED_PIECES if turn == 0 else BLK_PIECES
    for c in range(9):
        for r in range(10):
            if board_arr[c, r] in own:
                for tc, tr in get_pseudo_legal_moves(board_arr, c, r):
                    moves.append((c, r, tc, tr))
    return moves


def is_checkmate(board_arr, turn):
    """判断 turn 方是否被将死或困毙（无任何合法走法）"""
    own = RED_PIECES if turn == 0 else BLK_PIECES
    for c in range(9):
        for r in range(10):
            if board_arr[c, r] in own:
                for tc, tr in get_pseudo_legal_moves(board_arr, c, r):
                    # 模拟走子，看是否仍被将军
                    tmp = board_arr.copy()
                    tmp[tc, tr] = tmp[c, r]
                    tmp[c, r] = EMPTY
                    if not is_in_check(tmp, turn):
                        return False  # 存在合法走法
    return True  # 无合法走法 → 将死或困毙


# ===================== 加载训练好的模型 =====================
def load_trained_model():
    model = HybridXiangqiModel().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()  # 推理模式
    print("✅ 最优模型加载成功！")
    return model


# ===================== 核心：AI 走子推理（和训练100%对齐） =====================
@torch.no_grad()
def ai_move(model, board):
    # 1. 重建状态（完全复用训练代码的reconstruct_state）
    valid_history = [m for m in board.history[-HISTORY_LEN:]]
    state = reconstruct_state(board.board, valid_history, board.turn)

    # 2. 转换张量
    x = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)

    # 3. 模型预测
    pred_f, pred_t = model(x)

    # 4. 合法掩码（和训练完全一致：直接相加）
    m_f, m_t = get_legal_mask(board.board, board.turn)
    m_f = torch.tensor(m_f, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    m_t = torch.tensor(m_t, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 5. 掩码计算（和训练代码一模一样！）
    pred_f = pred_f + m_f
    pred_t = pred_t + m_t

    # 6. 选择概率最高的合法走子
    # 🔥 修正：先选 from，再在它的合法目标中选最优的 to（防止 from/to 不匹配的非法走子）
    pred_f_flat = pred_f.squeeze(0)  # (90,)
    pred_t_flat = pred_t.squeeze(0)  # (90,)

    # 按 from 得分排序，从高到低尝试
    from_scores, from_indices = torch.sort(pred_f_flat, descending=True)

    best_move = None
    best_score = -float("inf")
    for from_idx in from_indices:
        fid = from_idx.item()
        fc, fr = fid // 10, fid % 10

        # 获取该棋子的所有合法走法
        legal = get_pseudo_legal_moves(board.board, fc, fr)
        if not legal:
            continue

        # 在合法目标中选得分最高的
        for tc, tr in legal:
            tid = tc * 10 + tr
            score = pred_t_flat[tid].item()
            if score > best_score:
                best_score = score
                best_move = (fc, fr, tc, tr)

        # 只要当前 from_idx 有合法走法，就停止（score 已取该 from 下的最优 to）
        if best_move is not None:
            break

    if best_move is None:
        # 彻底无合法走子（将杀/困毙）
        return (0, 0, 0, 0)

    # ===================== 【仅新增这4行：规则兜底！被将军必须解将】 =====================
    fc, fr, tc, tr = best_move
    if is_in_check(board.board, board.turn):
        # 模拟走子，无效解将就重新选走法
        tmp_board = board.board.copy()
        tmp_board[tc, tr] = tmp_board[fc, fr]
        tmp_board[fc, fr] = EMPTY
        if is_in_check(tmp_board, board.turn):
            # 重新筛选：只选能解将的走法
            for move in get_all_legal_moves(board.board, board.turn):
                c_f, c_r, c_tc, c_tr = move
                tmp = board.board.copy()
                tmp[c_tc, c_tr] = tmp[c_f, c_r]
                tmp[c_f, c_r] = EMPTY
                if not is_in_check(tmp, board.turn):
                    return move
    # ==================================================================================

    return (fc, fr, tc, tr)


# ===================== 绘制棋盘 + 圆形棋子+居中中文 =====================
def draw_graphic_board(board_obj):
    # 填充背景
    screen.fill(BG_COLOR)

    # ── 水平线（10条，从左边界到右边界，全部连续） ──
    for i in range(10):
        y = i * CELL_SIZE + BOARD_PADDING
        pygame.draw.line(
            screen, LINE_COLOR, (BOARD_PADDING, y), (BOARD_WIDTH + BOARD_PADDING, y), 3
        )

    # ── 垂直线（9条）—— 内线（列1~7）不穿过河界 ──
    for i in range(9):
        x = i * CELL_SIZE + BOARD_PADDING
        if i == 0 or i == 8:
            # 左右边界线：贯通
            pygame.draw.line(
                screen,
                LINE_COLOR,
                (x, BOARD_PADDING),
                (x, BOARD_HEIGHT + BOARD_PADDING),
                3,
            )
        else:
            # 内线：分上下两段，中间断开为河界
            pygame.draw.line(
                screen,
                LINE_COLOR,
                (x, BOARD_PADDING),
                (x, 4 * CELL_SIZE + BOARD_PADDING),
                3,
            )
            pygame.draw.line(
                screen,
                LINE_COLOR,
                (x, 5 * CELL_SIZE + BOARD_PADDING),
                (x, BOARD_HEIGHT + BOARD_PADDING),
                3,
            )

    # ── 绘制楚河汉界 ──
    # 清空河界区域（覆盖掉之前画的竖线中间段）
    pygame.draw.rect(
        screen,
        BG_COLOR,
        (BOARD_PADDING, 4 * CELL_SIZE + BOARD_PADDING, BOARD_WIDTH, CELL_SIZE),
    )

    # ── 绘制九宫斜线 ──
    # 黑方九宫（上方）：(3,0)-(5,2) 和 (5,0)-(3,2)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (3 * CELL_SIZE + BOARD_PADDING, BOARD_PADDING),
        (5 * CELL_SIZE + BOARD_PADDING, 2 * CELL_SIZE + BOARD_PADDING),
        2,
    )
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (5 * CELL_SIZE + BOARD_PADDING, BOARD_PADDING),
        (3 * CELL_SIZE + BOARD_PADDING, 2 * CELL_SIZE + BOARD_PADDING),
        2,
    )
    # 红方九宫（下方）：(3,7)-(5,9) 和 (5,7)-(3,9)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (3 * CELL_SIZE + BOARD_PADDING, 7 * CELL_SIZE + BOARD_PADDING),
        (5 * CELL_SIZE + BOARD_PADDING, 9 * CELL_SIZE + BOARD_PADDING),
        2,
    )
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (5 * CELL_SIZE + BOARD_PADDING, 7 * CELL_SIZE + BOARD_PADDING),
        (3 * CELL_SIZE + BOARD_PADDING, 9 * CELL_SIZE + BOARD_PADDING),
        2,
    )

    # 加载中文字体（用于棋子）
    FONT_CANDIDATES = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    chess_font = None
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            chess_font = pygame.freetype.Font(fp, 24)
            break
    if chess_font is None:
        chess_font = pygame.freetype.SysFont("stheitilight", 24)

    # 绘制合法落点高亮 - 仅新增，不改动原有绘制
    for c, r in legal_moves:
        cx = c * CELL_SIZE + BOARD_PADDING
        cy = r * CELL_SIZE + BOARD_PADDING
        # 鼠标悬停的合法落点使用更亮的颜色
        h_col = round((mouse_x - BOARD_PADDING) / CELL_SIZE)
        h_row = round((mouse_y - BOARD_PADDING) / CELL_SIZE)
        h_col = max(0, min(8, h_col))
        h_row = max(0, min(9, h_row))
        if c == h_col and r == h_row:
            pygame.draw.rect(
                screen, (0, 200, 255), (cx - 25, cy - 25, 50, 50), 4
            )  # 青色粗边框
        else:
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, (cx - 25, cy - 25, 50, 50), 3)

    # ===================== 圆形棋子 + 居中中文文字 =====================
    for col in range(9):
        for row in range(10):
            piece = board_obj.board[col][row]
            if piece == EMPTY:
                continue

            # 计算棋子中心坐标（置于交叉点上）
            x = col * CELL_SIZE + BOARD_PADDING
            y = row * CELL_SIZE + BOARD_PADDING
            radius = 22

            # 鼠标悬停/选中高亮 - 仅新增，不改动原有棋子绘制
            h_col = round((mouse_x - BOARD_PADDING) / CELL_SIZE)
            h_row = round((mouse_y - BOARD_PADDING) / CELL_SIZE)
            h_col = max(0, min(8, h_col))
            h_row = max(0, min(9, h_row))
            if (col == selected_col and row == selected_row) or (
                col == h_col
                and row == h_row
                and board_obj.board[col][row] in RED_PIECES
            ):
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, (x - 25, y - 25, 50, 50), 3)

            # 如果当前棋子被选中拖动中，跳过原位绘制（后面会画在鼠标位置）
            if selected_col == col and selected_row == row:
                continue

            # 1. 绘制圆形棋子底色
            if piece in [RED_K, RED_R, RED_N, RED_C, RED_E, RED_A, RED_P]:
                pygame.draw.circle(screen, RED_CHESS, (x, y), radius)
                text_color = WHITE  # 红棋白字
            else:
                pygame.draw.circle(screen, BLACK_CHESS, (x, y), radius)
                text_color = WHITE  # 黑棋白字
            pygame.draw.circle(screen, WHITE, (x, y), radius, 2)

            # 2. 绘制居中的中文棋子文字
            text = CHESS_TEXT[piece]
            text_rect = chess_font.get_rect(text)
            # 计算文字居中偏移
            text_x = x - text_rect.width // 2
            text_y = y - text_rect.height // 2
            chess_font.render_to(screen, (text_x, text_y), text, text_color)

    # 右侧提示栏
    font_tip = None
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            font_tip = pygame.freetype.Font(fp, 30)
            break
    if font_tip is None:
        font_tip = pygame.freetype.SysFont("stheitilight", 30)

    # ── 绘制行列编号提示 ──
    label_font = (
        pygame.freetype.Font("/System/Library/Fonts/STHeiti Light.ttc", 22)
        if os.path.exists("/System/Library/Fonts/STHeiti Light.ttc")
        else pygame.freetype.SysFont("stheitilight", 22)
    )

    # 列标签 A-I（放在棋盘下方）
    for col_idx in range(9):
        label = chr(ord("A") + col_idx)
        lx = col_idx * CELL_SIZE + BOARD_PADDING
        ly = BOARD_HEIGHT + BOARD_PADDING + 30
        rect = label_font.get_rect(label)
        label_font.render_to(screen, (lx - rect.width // 2, ly), label, LINE_COLOR)

    # 行标签 0-9（放在棋盘左侧，9=黑方底线在上方，0=红方底线在下方）
    for row_idx in range(10):
        label = str(9 - row_idx)  # 用户输入的行号
        lx = BOARD_PADDING - 32
        ly = row_idx * CELL_SIZE + BOARD_PADDING
        rect = label_font.get_rect(label)
        label_font.render_to(
            screen, (lx - rect.width // 2, ly - rect.height // 2), label, LINE_COLOR
        )

    tips = ["红方：你", "黑方：AI", "走子格式：E7-E5", "关闭窗口退出"]
    y_pos = 50
    for tip in tips:
        font_tip.render_to(
            screen, (BOARD_WIDTH + BOARD_PADDING * 2 + 20, y_pos), tip, BLACK_CHESS
        )
        y_pos += 40

    # 拖拽棋子跟随鼠标 - 仅新增
    if selected_col != -1:
        piece = board_obj.board[selected_col][selected_row]
        x, y = mouse_x, mouse_y
        radius = 22
        if piece in RED_PIECES:
            pygame.draw.circle(screen, RED_CHESS, (x, y), radius)
        else:
            pygame.draw.circle(screen, BLACK_CHESS, (x, y), radius)
        pygame.draw.circle(screen, WHITE, (x, y), radius, 2)
        text = CHESS_TEXT[piece]
        text_rect = chess_font.get_rect(text)
        chess_font.render_to(
            screen, (x - text_rect.width // 2, y - text_rect.height // 2), text, WHITE
        )

    pygame.display.update()


# ===================== 人机对战主循环 - 仅替换命令行为鼠标，其余100%原样 =====================
def play_game():
    global selected_col, selected_row, legal_moves, mouse_x, mouse_y, GAME_OVER, WINNER_MSG, GAME_OVER_COLOR
    model = load_trained_model()
    board = XiangqiBoard()
    # 初始棋盘FEN（和训练一致）
    board.load_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    )

    print("🎮 象棋AI对战开始！你执红，AI执黑")
    print("鼠标操作：左键选子，左键落子，右键/ESC取消")
    GAME_OVER = False
    WINNER_MSG = ""
    running = True

    while running:
        clock.tick(60)
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # 监听窗口关闭
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

            # 鼠标移动
            if event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos

            # ESC取消选中（如果游戏结束则退出）
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if GAME_OVER:
                        running = False
                        pygame.quit()
                        return
                    selected_col, selected_row = -1, -1
                    legal_moves = []

            # 游戏结束后不再响应落子操作
            if GAME_OVER:
                continue

            # 鼠标点击事件
            if event.type == pygame.MOUSEBUTTONDOWN and board.turn == 0:
                mx, my = event.pos
                col = round((mx - BOARD_PADDING) / CELL_SIZE)
                row = round((my - BOARD_PADDING) / CELL_SIZE)
                col = max(0, min(8, col))
                row = max(0, min(9, row))

                # 右键取消（pygame中button=3是右键）
                if event.button == 3:
                    selected_col, selected_row = -1, -1
                    legal_moves = []
                
                # 左键选子/落子
                if event.button == 1:
                    if selected_col == -1:
                        # 选中红方棋子
                        if board.board[col][row] in RED_PIECES:
                            selected_col, selected_row = col, row
                            legal_moves = get_pseudo_legal_moves(board.board, col, row)
                    else:
                        # 合法落点落子
                        if (col, row) in legal_moves:
                            board.push((selected_col, selected_row, col, row))
                            selected_col, selected_row = -1, -1
                            legal_moves = []
                            # 落子后检查黑方（AI）是否被将死
                            if is_checkmate(board.board, 1):
                                GAME_OVER = True
                                WINNER_MSG = "🎉 红方胜！你赢了！"
                                GAME_OVER_COLOR = RED_CHESS
                                print("🎉 红方胜！你赢了！")

        # 刷新图形棋盘
        draw_graphic_board(board)

        # 游戏结束后不再走子
        if GAME_OVER:
            continue

        # 黑方（AI）走子
        if board.turn == 1:
            # 清除残留在界面上的人类上次的合法走法提示
            legal_moves = []
            print("🤖 AI思考中...")
            move = ai_move(model, board)
            fc, fr, tc, tr = move
            # 检测AI是否无棋可走（将杀/困毙 sentinel）
            if fc == 0 and fr == 0 and tc == 0 and tr == 0:
                GAME_OVER = True
                WINNER_MSG = "🎉 红方胜！你赢了！"
                GAME_OVER_COLOR = RED_CHESS
                print("🎉 AI无合法走法，红方胜！")
                continue
            col_map_rev = {v: k for k, v in COL_MAP.items()}
            print(f"AI走子：{col_map_rev[fc]}{fr} -> {col_map_rev[tc]}{tr}")
            board.push(move)
            # AI走完后检查红方（人类）是否被将死
            if is_checkmate(board.board, 0):
                GAME_OVER = True
                WINNER_MSG = "😵 黑方胜！AI赢了..."
                GAME_OVER_COLOR = BLACK_CHESS
                print("😵 黑方胜！AI赢了...")

        # 刷新棋盘（第二次刷新以显示AI走完的状态）
        draw_graphic_board(board)

if __name__ == "__main__":
    play_game()
