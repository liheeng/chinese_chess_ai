def predict_next_move(qipu_steps: list[str]):
    model = HybridXiangqiModel()
    model.load_state_dict(torch.load("hybrid_xiangqi.pth", map_location="cpu"))
    model.eval()

    board = xq.Board()
    for step in qipu_steps:
        board.push(xq.Move.from_uci(step))

    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        f_logits, t_logits = model(tensor)
        f_idx = torch.argmax(f_logits).item()
        t_idx = torch.argmax(t_logits).item()

    from_col, from_row = f_idx // 10, f_idx % 10
    to_col, to_row = t_idx // 10, t_idx % 10
    move = xq.Move(xq.Square(from_col, from_row), xq.Square(to_col, to_row))

    if move in board.legal_moves:
        return f"推荐走法：{move.uci()}（合法）"
    else:
        return f"预测走法非法，推荐最优合法走法：{list(board.legal_moves)[0]}"

# 测试
history = ["h2e2", "h8e8", "h0h2"]
print(predict_next_move(history))