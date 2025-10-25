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
    print("PyTorch Lightning + Hydra (Optimizer/Scheduler YAML対応)")
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
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    print("\n✅ DataModule作成完了")
    
    # ========== Optimizer/Scheduler設定を渡す ==========
    model = instantiate(
        cfg.model, optimizer_config=OmegaConf.to_container(
            cfg.optimizer, resolve=True
        ),
        scheduler_config=OmegaConf.to_container(
            cfg.scheduler, resolve=True
        ) if "scheduler" in cfg else None,
        class_names=datamodule.class_names
    )
    
    print(f"✅ Optimizer: {cfg.optimizer.name}")
    print(f"✅ Scheduler: {cfg.scheduler.name if 'scheduler' in cfg else 'None'}")
    
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


if __name__ == "__main__":
    main()
