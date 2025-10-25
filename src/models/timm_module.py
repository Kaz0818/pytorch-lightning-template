"""
timmベースのLightningModule（転移学習対応）
Classification Report & Confusion Matrix対応
"""
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


class TimmLitModule(pl.LightningModule):
    """
    timmベースのLightningModule
    - 転移学習（transfer learning）対応
    - ファインチューニング（fine-tuning）対応
    - Classification Report & Confusion Matrix対応
    """
    
    def __init__(
        self, 
        model_name="efficientnet_b0",
        num_classes=10, 
        lr=1e-3,
        pretrained=True,
        freeze_backbone=False,
        class_names=None, #クラス名のリスト（オプション)
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
        
        # テスト用の予測と・ラベル保存
        self.test_preds = []
        self.test_labels = []
    
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
        
        # 予測とラベルを保存
        self.test_preds.append(preds)
        self.test_labels.append(y)
        
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        
        return loss
    
    # ========== テスト終了時に呼ばれる ==========
    def on_test_epoch_end(self):
        """ 
        テスト終わった後に呼ばれる
        Classification Report と Confusion Matrixを作成
        """
        # 全バッチの予測とラベルを結合
        all_preds = torch.cat(self.test_preds).cpu().numpy()
        all_labels = torch.cat(self.test_labels).cpu().numpy()
        
        # クラス名を取得
        class_names = self.hparams.class_names
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.hparams.num_classes)]
        
        # ========== Classification Report ==========
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        
        report_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            digits=4
        )
        print(report_report)
        
        # 辞書形式(W&B用)
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        # ========== W&Bに保存（正しい方法） ==========
        if self.logger:
            #1. Tableとして保存
            report_df = pd.DataFrame(report_dict).transpose()
            self.logger.experiment.log({
                "classification_report_table": wandb.Table(dataframe=report_df)
            })
            
            #2. 各クラスのメトリクスを個別に保存
            for class_name, metrics in report_dict.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    self.logger.experiment.log({
                        f"test_metrics/{class_name}/precision": metrics['precision'],
                        f"test_metrics/{class_name}/recall": metrics['recall'],
                        f"test_metrics/{class_name}/f1-score": metrics['f1-score'],
                        f"test_metrics/{class_name}/support": metrics['support'],
                    })
                    
            # 3. 全体のメトリクス
            self.logger.experiment.log({
                "test/accuracy": report_dict['accuracy'],
                "test/macro_avg/precision": report_dict['macro avg']['precision'],
                "test/macro_avg/recall": report_dict['macro avg']['recall'],
                "test/macro_avg/f1-score": report_dict['macro avg']['f1-score'],
                "test/weighted_avg/precision": report_dict['weighted avg']['precision'],
                "test/weighted_avg/recall": report_dict['weighted avg']['recall'],
                "test/weighted_avg/f1-score": report_dict['weighted avg']['f1-score'],
            })
            
            print("✅ Classification Report を W&B に保存完了")
                    
    
        # ========== Confusion Matrix ==========
        cm = confusion_matrix(all_labels, all_preds)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        if self.logger:
            # W&B専用のインタラクティブ版
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=class_names
                ),
                # Seabornのヒートマップ
                "confusion_matrix_heatmap": wandb.Image(fig)
            })
        
        fig.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # メモリクリア
        self.test_preds.clear()
        self.test_labels.clear()
            
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
