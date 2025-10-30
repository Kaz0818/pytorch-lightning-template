# Repository Guidelines

## プロジェクト構成とモジュール
- `configs/`：Hydra設定。`config.yaml`を基点に`model/`,`datamodule/`,`callbacks/`で上書き。新規構成は既存YAMLを参考に命名をスネークケースで追加。
- `src/`：実装本体。`models/`はLightningModule、`datamodules/`はLightningDataModule、`utils/`は共通処理。拡張時は型ヒントとdocstringで入出力を明示。
- `data/`,`outputs/`,`lightning_logs/`,`tb_logs/`,`wandb/`：生成物。大きな成果物はGit管理外を推奨し、再現に必要な設定のみを残す。
- `train.py` / `train_kaggle.py`：主要エントリーポイント。Kaggle実行ではKaggle固有のパスに合わせて`configs/datamodule/`を上書き。

## ビルド・テスト・ローカル実行コマンド
- `python -m venv .venv && source .venv/bin/activate`：仮想環境作成。
- `pip install -r requirements.txt`：依存関係インストール。
- `python train.py`：既定構成で学習とテスト。Hydraのオーバーライド例：`python train.py model=resnet18 trainer.max_epochs=20`。
- `python train.py trainer.fast_dev_run=true`：高速な健全性チェック。新しいモデルやデータ導入時は必須。
- `python train_kaggle.py`：Kaggle向け軽量実行。アウトプット送信前に`--smoke_test`等の独自フラグを追加する場合はREADME追記。
- `tensorboard --logdir=tb_logs`：ローカルメトリクス可視化。

## コーディングスタイルと命名
- Python 3系、4スペースインデント、PEP 8準拠。既存コードに倣いクラス名は`CamelCase`、関数・変数・Hydraのキーは`snake_case`。
- LightningModuleは`class_names`やハイパーパラメータを`__init__`で受け、`configure_optimizers`では`optimizer_config`/`scheduler_config`を尊重。
- 設定ファイルは「対象 + 用途」のスネークケース（例：`configs/model/efficientnet_b3.yaml`）。共通値は共通YAMLへ切り出し、`defaults`で再利用。

## テストガイドライン
- 迅速な回帰確認に`trainer.fast_dev_run=true`や`trainer.limit_train_batches=0.1`を活用。長時間ジョブは`outputs/`へログを残し再現条件を記録。
- 自動テスト追加時は`tests/`ディレクトリを作成し`pytest`を想定。テスト名は`test_<対象>_<条件>()`形式でデータモックを最小化。
- 新規DataModuleはバッチ整合性とクラス数検証の単体テストを追加し、`cfg.datamodule`の必須キーを明文化。

## コミットとプルリクエスト
- Git履歴はConventional Commits（例：`feat: add timm backbone wrapper`,`fix: adjust scheduler config`）。スコープは省略可だが主要モジュール名を推奨。
- プルリクでは概要、主な設定変更（Hydraオーバーライド例）、検証結果（メトリクス／ログパス）、既知のTODOを箇条書きで記載。
- 実験ログ共有時は`lightning_logs/`や`wandb/`のIDを記載し、再現手順を`python train.py ...`形式で残す。

## 設定とシークレット運用
- 機密キーは環境変数で注入し、Hydra `configs/logger/`ではプレースホルダを用意。`.env`を使用する場合は`.gitignore`へ登録。
- `configs/`変更時は`hydra.verbose`を一時的に有効化して差分を確認し、想定しない上書きがないか`python train.py --cfg job`で検証。
