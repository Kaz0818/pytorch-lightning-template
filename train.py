"""
PyTorch Lightning + Hydra トレーニングスクリプト
Optimizer/Scheduler YAML対応版
"""
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """メイン実行関数"""
    
    _print_header()
    _print_config(cfg)
    
    # DataModule
    datamodule = _setup_datamodule(cfg)
    
    # Model
    model = _setup_model(cfg, datamodule)
    
    # Logger & Callbacks
    logger = _setup_logger(cfg)
    callbacks = _setup_callbacks(cfg)
    
    # Trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    
    # 訓練 & テスト
    _run_training(trainer, model, datamodule)
    
    print("\n✅ 完了！")


def _print_header():
    """ヘッダー表示"""
    print("=" * 60)
    print("PyTorch Lightning + Hydra")
    print("=" * 60)


def _print_config(cfg: DictConfig):
    """設定表示"""
    print("\n【設定】")
    print(f"Model: {cfg.model.model_name}")
    print(f"Optimizer: {cfg.optimizer.name}")
    print(f"Scheduler: {cfg.scheduler.name if 'scheduler' in cfg else 'None'}")
    print(f"Max Epochs: {cfg.trainer.max_epochs}")


def _setup_datamodule(cfg: DictConfig):
    """DataModuleセットアップ"""
    datamodule = instantiate(cfg.datamodule, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    print(f"\n✅ DataModule: {len(datamodule.train_dataset)} train")
    return datamodule


def _setup_model(cfg: DictConfig, datamodule):
    """Modelセットアップ"""
    model = instantiate(
        cfg.model,
        optimizer_config=OmegaConf.to_container(cfg.optimizer, resolve=True),
        scheduler_config=OmegaConf.to_container(cfg.scheduler, resolve=True) if 'scheduler' in cfg else None,
        class_names=datamodule.class_names
    )
    print("✅ Model作成完了")
    return model


def _setup_logger(cfg: DictConfig):
    """Loggerセットアップ"""
    try:
        logger = instantiate(cfg.logger)
        print("✅ Logger作成完了")
        return logger
    except Exception as e:
        print(f"⚠️ Logger作成失敗: {e}")
        from pytorch_lightning.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir="tb_logs")


def _setup_callbacks(cfg: DictConfig):
    """Callbacksセットアップ"""
    callbacks = [
        instantiate(cfg.callbacks.checkpoint),
        instantiate(cfg.callbacks.early_stopping)
    ]
    print("✅ Callbacks作成完了")
    return callbacks


def _run_training(trainer, model, datamodule):
    """訓練とテストを実行"""
    print("\n【訓練開始】")
    trainer.fit(model, datamodule)
    
    print("\n【テスト開始】")
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
