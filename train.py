import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    
    # ========== Rank 0 だけで実行 ==========
    if pl.utilities.rank_zero_only.rank == 0:
        print("=" * 60)
        print("PyTorch Lightning + Hydra (Kaggle)")
        print("=" * 60)
        print("\n【読み込まれた設定】")
        print(OmegaConf.to_yaml(cfg))
    # =====================================
    
    # DataModule
    datamodule = instantiate(cfg.datamodule, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    if pl.utilities.rank_zero_only.rank == 0:
        print("\n✅ DataModule作成完了")
        print(f"   Train: {len(datamodule.train_dataset)} 枚")
        print(f"   Val: {len(datamodule.val_dataset)} 枚")
    
    # Model
    model = instantiate(
        cfg.model,
        optimizer_config=OmegaConf.to_container(cfg.optimizer, resolve=True),
        scheduler_config=OmegaConf.to_container(cfg.scheduler, resolve=True) if 'scheduler' in cfg else None,
        class_names=datamodule.class_names
    )
    
    if pl.utilities.rank_zero_only.rank == 0:
        print(f"✅ Model: {cfg.model.model_name}")
        print(f"✅ Optimizer: {cfg.optimizer.name}")
        print(f"✅ Scheduler: {cfg.scheduler.name if 'scheduler' in cfg else 'None'}")
    
    # Logger
    logger = instantiate(cfg.logger)
    
    # Callbacks
    checkpoint_callback = instantiate(cfg.callbacks.checkpoint)
    early_stop_callback = instantiate(cfg.callbacks.early_stopping)
    callbacks = [checkpoint_callback, early_stop_callback]
    
    # Trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    
    if pl.utilities.rank_zero_only.rank == 0:
        print("\n【訓練開始】")
    
    # 訓練
    trainer.fit(model, datamodule)
    
    if pl.utilities.rank_zero_only.rank == 0:
        print("\n【テスト開始】")
    
    # テスト
    trainer.test(model, datamodule, ckpt_path="best")
    
    if pl.utilities.rank_zero_only.rank == 0:
        print("\n✅ 完了！")


if __name__ == "__main__":
    main()