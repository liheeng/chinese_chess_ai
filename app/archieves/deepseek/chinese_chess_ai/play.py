import torch
from archieves.deepseek.chinese_chess_ai.core import Board
from archieves.deepseek.chinese_chess_ai.model import ChineseChessNet
from archieves.deepseek.chinese_chess_ai.iccs_loader import ICCSDataset  # 复用编码函数


class AIEngine:
    def __init__(self, model_path):
        self.model = ChineseChessNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.dataset_helper = ICCSDataset([])  # 只是为了用它的编码函数
        self.history = []

    def get_best_move(self, board):
        # 更新历史
        self.history.append(board.__class__())
        hist_board = Board()
        hist_board.board = [row[:] for row in board.board]
        hist_board.turn = board.turn
        self.history[-1] = hist_board
        if len(self.history) > 4:
            self.history.pop(0)
        state = self.dataset_helper._encode_state(self.history)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            probs = torch.softmax(logits[0], dim=-1).numpy()
        # 获取合法走法并选最高概率
        legal_moves = board.get_legal_moves()
        best_move = None
        best_prob = -1
        for (fx, fy), (tx, ty) in legal_moves:
            from_idx = fy * 9 + fx
            to_idx = ty * 9 + tx
            move_idx = from_idx * 90 + to_idx
            prob = probs[move_idx]
            if prob > best_prob:
                best_prob = prob
                best_move = ((fx, fy), (tx, ty))
        return best_move


def main():
    engine = AIEngine("chess_model.pth")
    board = Board()
    print(
        "中国象棋 AI 对战，红方（下）你先走。输入走法格式：x1y1 x2y2，例如 4 9 4 7 表示炮二平五"
    )
    while True:
        print(board)
        if board.turn == 0:
            move_str = input("红方: ")
            if move_str == "quit":
                break
            parts = move_str.split()
            if len(parts) == 4:
                fx, fy, tx, ty = map(int, parts)
                if board.is_legal_move((fx, fy), (tx, ty)):
                    board.push((fx, fy), (tx, ty))
                else:
                    print("非法走法")
            else:
                print("格式错误，请输入四个数字: x1 y1 x2 y2")
        else:
            print("AI 思考中...")
            move = engine.get_best_move(board)
            if move:
                board.push(*move)
                print(f"AI 走: {move}")
            else:
                print("AI 无步可走")
                break
        if board.result() != "*":
            print(f"对局结束: {board.result()}")
            break


if __name__ == "__main__":
    main()
