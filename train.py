"""
Step 6c: defaults + _target_ を使った完全構成
- Hydraのinstantiateで自動オブジェクト生成
- defaults で設定ファイルを組み合わせ
- コマンドラインで簡単に切り替え
"""
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    メイン実行関数
    
    Hydraのinstantiateで全てのオブジェクトを自動生成
    """
    
    print("=" * 60)
    print("Step 6c: Hydra完全統合（defaults + _target_）")
    print("=" * 60)
    
    # 設定を表示
    print("\n【読み込まれた設定】")
    print(OmegaConf.to_yaml(cfg))
    
    # ==========================================================
    # Hydra instantiate の魔法
    # _target_ を持つ設定は自動的にオブジェクト化される
    # ==========================================================
    
    # DataModule（_target_ から自動生成）
    datamodule = instantiate(cfg.datamodule, _recursive_=False)
    print("\n✅ DataModule作成完了")
    
    # Model（_target_ から自動生成）
    model = instantiate(cfg.model)
    print("✅ Model作成完了")
    
    # Logger（_target_ から自動生成）
    try:
        logger = instantiate(cfg.logger)
        print(f"✅ Logger作成完了: {cfg.logger._target_.split('.')[-1]}")
    except Exception as e:
        print(f"⚠️  Logger作成失敗: {e}")
        print("TensorBoardにフォールバック")
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir="tb_logs", name="fallback")
    
    # Callbacks（_target_ から自動生成）
    checkpoint_callback = instantiate(cfg.callbacks.checkpoint)
    early_stop_callback = instantiate(cfg.callbacks.early_stopping)
    callbacks = [checkpoint_callback, early_stop_callback]
    print("✅ Callbacks作成完了")
    
    # Trainer（設定から作成、callbacksとloggerを渡す）
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    print("✅ Trainer作成完了")
    
    # 訓練
    print("\n【訓練開始】")
    trainer.fit(model, datamodule)
    
    # テスト
    print("\n【テスト開始】")
    trainer.test(model, datamodule, ckpt_path="best")
    
    print("\n" + "=" * 60)
    print("✅ Step 6c完了！PyTorch Lightning完全マスター！")
    print("=" * 60)
    print("\n学んだこと:")
    print("- defaults: 複数の設定ファイルを組み合わせ")
    print("- _target_: クラスパスを指定して自動インスタンス化")
    print("- instantiate(): _target_ を持つ設定からオブジェクト生成")
    print("- 設定ファイルだけで全てを管理")
    print("- コマンドラインで簡単に切り替え")
    
    print("\n便利なコマンド例:")
    print("  # 基本実行（W&B使用）")
    print("  python train_final.py")
    print("\n  # TensorBoardに切り替え")
    print("  python train_final.py logger=tensorboard")
    print("\n  # モデルを変更")
    print("  python train_final.py model=resnet18")
    print("  python train_final.py model=efficientnet_b3")
    print("\n  # 転移学習モード")
    print("  python train_final.py model.freeze_backbone=true model.lr=0.001")
    print("\n  # バッチサイズとエポック数を変更")
    print("  python train_final.py datamodule.batch_size=64 trainer.max_epochs=20")
    print("\n  # 複数の設定を組み合わせ")
    print("  python train_final.py model=resnet18 logger=tensorboard trainer.max_epochs=5")
    print("\n  # 実験名を設定")
    print("  python train_final.py experiment_name=exp001_resnet18")


if __name__ == "__main__":
    main()
