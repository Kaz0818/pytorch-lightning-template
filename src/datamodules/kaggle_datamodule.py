""" 
Kaggle DatasetでのDataModuleコード
"""
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder

class KaggleDataModule(pl.LightningDataModule):
    """ 
    Kaggle Plant Disease Recognition Dataset用DataModule
    同じようなDataset構造なら流用できる
    - 既にtrain/val/testが分かれているので層化分割不要
    - ImageFolderを使用
    """
    
    def __init__(self,
                 data_dir="/kaggle/input/plant-disease-recognition-dataset", # ここはデータが有る場所を指定
                 batch_size=64,
                 num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #データパスを定義(インスタンス変数として保持)
        self.train_dir = Path(self.hparams.data_dir) / "Train" / "Train"
        self.val_dir = Path(self.hparams.data_dir) / "Validation" / "Validation"
        self.test_dir = Path(self.hparams.data_dir) / "Test" / "Test"

        self.class_names = None
        
        # 訓練用の拡張（Data Augmentation）
        self.train_transform = transforms.Compose([
            transforms.Resize(256),                    # まずリサイズ
            transforms.RandomCrop(224),                # ランダムクロップ
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet標準
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 検証・テスト用（拡張なし）
        self.val_transform = transforms.Compose([
            transforms.Resize(256),                    # まずリサイズ
            transforms.CenterCrop(224),                # 中央クロップ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self):
        """ 
        データのダウンロード(1回のみ実行)
        Kaggleデータセットは既に存在するので、存在確認のみ
        """
        # データディレクトリの存在確認
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not self.val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        print(f"✅️データディレクトリ確認完了")

        
    def setup(self, stage=None):
        """データセットの作成"""
        if stage == "fit" or stage is None:
            # 訓練データセット(Data Augmentation適用)
            self.train_dataset = ImageFolder(root=self.train_dir,
                                             transform=self.train_transform)
            # 検証データセット(拡張なし)
            self.val_dataset = ImageFolder(root=self.val_dir,
                                           transform=self.val_transform)

            self.class_names = self.train_dataset.classes
            
            print(f"✅ データセット作成完了:")
            print(f"   Train: {len(self.train_dataset)}枚")
            print(f"   Val:   {len(self.val_dataset)}枚")
            print(f"   Classes: {len(self.train_dataset.classes)}")            
            
            
            
        if stage == "test" or stage is None:
            # テストデータセット(拡張なし)
            self.test_dataset = ImageFolder(
                root=self.test_dir,
                transform=self.val_transform
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
