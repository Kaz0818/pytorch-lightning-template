"""
STL-10データセット用DataModule (train_val分割対応)
"""
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from torchvision import transforms
from sklearn.model_selection import train_test_split

class STL10DataModule(pl.LightningDataModule):
    """STL-10データセット用DataModule（Data Augmentation対応）
    STL-10データセット用DataModule
    - trainデータをtrain/valに層化分割
    - testは元々のtest splitを使用
    """
    
    def __init__(self,
                 data_dir="./data",
                 batch_size=32,
                 num_workers=4,
                 val_ratio=0.2,   # 検証データの割合
                 seed=42,         # 再現性のためのシード
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 訓練用の拡張（Data Augmentation）
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                               std=[0.2603, 0.2566, 0.2713])
        ])
        
        # 検証・テスト用（拡張なし）
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                               std=[0.2603, 0.2566, 0.2713])
        ])
    
    def prepare_data(self):
        """データのダウンロード（1回のみ実行）"""
        STL10(root=self.hparams.data_dir, split="train", download=True)
        STL10(root=self.hparams.data_dir, split="test", download=True)
    
    def setup(self, stage=None):
        """データセットの作成"""
        if stage == "fit" or stage is None:
            # ラベル取得用(transformなし)
            raw_dataset = STL10(
                root=self.hparams.data_dir,
                split="train",
                download=False
            )
            
            # ラベルを取得
            labels = np.array(raw_dataset.labels)
            indices = np.arange(len(raw_dataset))
            
            # train/val分割 (層化対応)
            train_indices, val_indices = train_test_split(
                indices,
                test_size=self.hparams.val_ratio,
                random_state=self.hparams.seed,
                stratify=labels   # <- 層化分割
            )
            
            print(f"✅ データ分割完了:")
            print(f"   Train: {len(train_indices)}枚")
            print(f"   Val:   {len(val_indices)}枚")
            
            # 訓練データセット(Data Augmentation適用)
            train_full = STL10(
                root=self.hparams.data_dir,
                split="train",
                transform=self.train_transform,
                download=False
            )
            self.train_dataset = Subset(train_full, train_indices)

            # 検証データセット(拡張なし)
            val_full = STL10(
                root=self.hparams.data_dir,
                split='train', # <= trainから分割
                transform=self.val_transform,
                download=False
            )
            self.val_dataset = Subset(val_full, val_indices)
            
        if stage == "test" or stage is None:
            # テストデータセット(元々のtest split)
            self.test_dataset = STL10(
                root=self.hparams.data_dir,
                split="test",  # <= 別のtest split
                transform=self.val_transform,
                download=False
            )
            print(f"✅ テストデータ: {len(self.test_dataset)}枚")
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
