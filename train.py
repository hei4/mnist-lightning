import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from models.multi_task_lightning_module import MultiTaskLightningModule
from data.multi_task_mnist_module import MutlTaskMnistModule

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # setup random seed
    L.seed_everything(cfg.seed, workers=True)

    # init the model module
    model_module = MultiTaskLightningModule(train=True)

    # setup data module
    data_module = MutlTaskMnistModule(
        cfg.data_module.root,
        train=True,
        thresold=cfg.data_module.threshold,
        sigma=cfg.data_module.sigma,
        seed=cfg.seed
    )

    # setup logger
    mlflow_logger = MLFlowLogger(experiment_name="lightning_logs")

    # setup Trainer
    trainer = L.Trainer(
        fast_dev_run=cfg.trainer.fast_dev_run,
        max_epochs=cfg.trainer.max_epochs,
        logger=mlflow_logger
    )
    
    # fitting
    trainer.fit(model=model_module, datamodule=data_module)

if __name__ == "__main__":
    main()