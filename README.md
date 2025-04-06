# mnist-lightning

MNIST + PyTorch Lightning + MLFlow + Hydra

## 環境設定

> Python 3.8.19
>
> torch               1.12.1+cu113
> torchvision         0.13.1+cu113
>
> lightning	2.1.4
> mlflow		2.17.2

```bash
pip install lightning[extra] torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mlflow
pip install torchmetrics[detection]
```

## 公式チュートリアル

[Lightning in 15 minutes](https://lightning.ai/docs/pytorch/2.1.4/starter/introduction.html)

- Level 1: Train a model
  - [Train a model (basic)](https://lightning.ai/docs/pytorch/2.1.4/model/train_model_basic.html)
- Level 2: Add a validation and test set
  - [Validate and test a model (basic)](https://lightning.ai/docs/pytorch/2.1.4/common/evaluation_basic.html)
  - [Saving and loading checkpoints (basic)](https://lightning.ai/docs/pytorch/2.1.4/common/checkpointing_basic.html)
- Level 5: Debug, visualize and find performance bottlenecks
  - [Debug your model (basic)](https://lightning.ai/docs/pytorch/2.1.4/debug/debugging_basic.html)
  - [Track and Visualize Experiments (basic)](https://lightning.ai/docs/pytorch/2.1.4/visualize/logging_basic.html)
- Level 6: Predict with your model
  - [Deploy models into production (basic)](https://lightning.ai/docs/pytorch/2.1.4/deploy/production_basic.html)
  - [Deploy models into production (intermediate)](https://lightning.ai/docs/pytorch/2.1.4/deploy/production_intermediate.html)

- Level 8: Modularize your projects
  - [LightningDataModule](https://lightning.ai/docs/pytorch/2.1.4/data/datamodule.html)

- Level 9: Understand your model
  - [Track and Visualize Experiments (intermediate)](https://lightning.ai/docs/pytorch/2.1.4/visualize/logging_intermediate.html)

## TorchMetrics

[TorchMetrics in PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)

## 参考になる情報

[pytorch-lightningの使い方](https://qiita.com/shibaura/items/b835fb61708dc52bb6b3)

[PyTorch Lightning 2021 (for MLコンペ)](https://qiita.com/fam_taro/items/df8656a6c3b277f58781)

[PyTorch Lightning の API を勉強しよう](https://qiita.com/ground0state/items/c1d705ca2ee329cdfae4)

[ヘビーユーザーが解説するPyTorch Lightning](https://tech.jxpress.net/entry/2021/11/17/112214)

## MLFlow + Hydra

[Hydra + PyTorch Lightningを使ったDeep Learning モデル構築テンプレート紹介](https://zenn.dev/mixi/articles/13b8cf80afcd93)

[設定管理ツール Hydra で内部構造ごと書き換える。](https://zenn.dev/gesonanko/articles/417d43669cf2af)

[Mnist_PytorchLightning_Hydra_Mlflow_Optuna](https://github.com/ryuseiasumo/Mnist_PytorchLightning_Hydra_Mlflow_Optuna)

[Hydra + MLFlow sample framework based on PyTorch-Lightning](https://github.com/k4noinfo/PytorchLightning_Hydra_MLFlow_Optuna)

[Pytorch Lightning Template using Hydra and MLFlow](https://github.com/kredde/pytorch-lightning-hydra-mlflow)

[MLflowとHydraを利用した実験管理](https://speakerdeck.com/futabato/mlflowtohydrawoli-yong-sitashi-yan-guan-li)

[Hydra, MLflow, Optunaの組み合わせで手軽に始めるハイパーパラメータ管理](https://supikiti22.medium.com/hydra-mlflow-optuna%E3%81%AE%E7%B5%84%E3%81%BF%E5%90%88%E3%82%8F%E3%81%9B%E3%81%A7%E6%89%8B%E8%BB%BD%E3%81%AB%E5%A7%8B%E3%82%81%E3%82%8B%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E7%AE%A1%E7%90%86-6b8e6d41b3da)

[MLflow+Hydra 最小構成](https://zenn.dev/kot/articles/fbda0c015069c2)
