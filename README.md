# PyTorch Lightning + Hydra Template

STL-10データセットでの画像分類を例にした、PyTorch Lightning + Hydra構成のテンプレート。

## 🚀 Features

- ✅ PyTorch Lightning による簡潔な学習コード
- ✅ Hydra による柔軟な設定管理
- ✅ W&B / TensorBoard による実験追跡
- ✅ timm による事前学習モデル統合
- ✅ 転移学習 & ファインチューニング対応
- ✅ Kaggle / Colab 対応

## 📁 Project Structure
lightning_tutorial/
├── configs/ # Hydra設定ファイル
│ ├── config.yaml # メイン設定
│ ├── model/ # モデル設定
│ ├── datamodule/ # データ設定
│ ├── logger/ # Logger設定
│ └── callbacks/ # Callback設定
├── src/ # ソースコード
│ ├── models/ # モデル定義
│ ├── datamodules/ # DataModule定義
│ └── utils/ # ユーティリティ
├── train_final.py # 訓練スクリプト
├── requirements.txt
└── README.md


## 🔧 Installation

### Local
git clone https://github.com/your-username/lightning-tutorial.git
cd lightning-tutorial
pip install -r requirements.txt


### Kaggle Notebook
!git clone https://github.com/your-username/lightning-tutorial.git
%cd lightning-tutorial
!pip install -r requirements.txt


### Google Colab
!git clone https://github.com/your-username/lightning-tutorial.git
%cd lightning-tutorial
!pip install -r requirements.txt


## 🎯 Quick Start

### 基本実行
python train_final.py


### モデルを変更
python train_final.py model=resnet18
python train_final.py model=efficientnet_b3


### Loggerを切り替え
TensorBoard
python train_final.py logger=tensorboard

W&B
python train_final.py logger=wandb


### 複数設定を変更
python train_final.py model=resnet18 logger=tensorboard trainer.max_epochs=20


### 転移学習モード
python train_final.py model.freeze_backbone=true model.lr=0.001


## 📊 Experiment Tracking

### TensorBoard
tensorboard --logdir=tb_logs


### W&B

実行時に自動的にブラウザが開きます。

## 🎓 Learning Steps

このプロジェクトは以下のステップで構築されました：

- **Step 1**: 最小構成のPyTorch Lightning
- **Step 2**: DataModuleの導入
- **Step 3**: Callbacks（EarlyStopping, ModelCheckpoint）
- **Step 4**: Logger（TensorBoard, W&B）
- **Step 5**: timm統合（転移学習）
- **Step 6**: Hydra統合とモジュール化

## 🔨 Customization

### 新しいモデルを追加
configs/model/your_model.yaml
target: src.models.timm_module.TimmLitModule
model_name: your_model_name
num_classes: 10
lr: 0.001
pretrained: true


### 新しいデータセットを追加
src/datamodules/your_dataset_datamodule.py
class YourDataModule(pl.LightningDataModule):
# 実装
configs/datamodule/your_dataset.yaml
target: src.datamodules.your_dataset_datamodule.YourDataModule
data_dir: ./data
batch_size: 32

# 使い方
!python train.py \
    datamodule=kaggle_plant \
    model=efficientnet_b0 \
    model.num_classes=3 \
    model.lr=0.0003 \
    trainer.accelerator=gpu \
    trainer.devices=2 \
    trainer.strategy=ddp \
    trainer.max_epochs=1 \
    experiment_name="efficientnet_b0_base"


## 📝 License

MIT License

## 🙏 Acknowledgments

- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [Hydra](https://github.com/facebookresearch/hydra)
- [timm](https://github.com/huggingface/pytorch-image-models)








