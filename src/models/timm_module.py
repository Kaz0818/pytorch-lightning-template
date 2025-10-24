"""
timmベースのLightningModule（転移学習対応）
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm


class TimmLitModule(pl.LightningModule):
    """
    timmベースのLightningModule
    - 転移学習（transfer learning）対応
    - ファインチューニング（fine-tuning）対応
    """
    
    def __init__(
        self, 
        model_name="efficientnet_b0",
        num_classes=10, 
        lr=1e-3,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # timmからモデルを作成
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        
        # 転移学習: バックボーンを凍結
        if freeze_backbone:
            self._freeze_backbone()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _freeze_backbone(self):
        """バックボーン（特徴抽出部分）を凍結"""
        print("🔒 バックボーンを凍結（転移学習モード）")
        
        for name, param in self.model.named_parameters():
            if not any(x in name for x in ['classifier', 'head', 'fc']):
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"訓練可能パラメータ: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc",
            }
        }
