import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from archieves.deepseek.chinese_chess_ai.iccs_loader import ICCSDataset
from archieves.deepseek.chinese_chess_ai.model import ChineseChessNet


class ChessLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        state, move_idx, value = batch
        # 注意：state 已经是 numpy 数组，需要转为 tensor
        state = torch.FloatTensor(state)
        move_idx = torch.LongTensor(move_idx)
        value = torch.FloatTensor(value)
        policy, value_pred = self.model(state)
        loss_p = self.cross_entropy(policy, move_idx)
        loss_v = self.mse(value_pred, value)
        loss = loss_p + loss_v
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        state, move_idx, value = batch
        state = torch.FloatTensor(state)
        move_idx = torch.LongTensor(move_idx)
        value = torch.FloatTensor(value)
        policy, value_pred = self.model(state)
        loss_p = self.cross_entropy(policy, move_idx)
        loss_v = self.mse(value_pred, value)
        loss = loss_p + loss_v
        self.log("val_loss", loss)
        acc = (policy.argmax(dim=1) == move_idx).float().mean()
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


if __name__ == "__main__":
    # 请将你的 ICCS 文件路径填入
    # PGN_PATH = "./iccs_lib/ICCS-99813/dpxq-99813games.pgns"
    PGN_PATH = "./iccs_lib/ICCS-41743/WXF-41743games.pgns"
    dataset = ICCSDataset([PGN_PATH], max_games=5000)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # 修正：使用导入的 random_split
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    model = ChineseChessNet()
    lit_model = ChessLightning(model)
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
    )
    trainer.fit(lit_model, train_loader, val_loader)
    torch.save(model.state_dict(), "chess_model.pth")
