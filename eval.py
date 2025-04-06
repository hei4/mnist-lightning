import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L

from models.multi_task_lightning_module import MultiTaskLightningModule
from data.multi_task_mnist_module import MutlTaskMnistModule

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # initialize model
    model_module = MultiTaskLightningModule.load_from_checkpoint("lightning_logs/version_18/checkpoints/epoch=9-step=1000.ckpt")

    # disable randomness, dropout, etc...
    model_module.eval()

    # setup data module
    data_module = MutlTaskMnistModule(
        cfg.data_module.root,
        train=False,
        thresold=cfg.data_module.threshold,
        sigma=cfg.data_module.sigma
    )

    # initialize the Trainer
    trainer = L.Trainer()

    # test the model
    trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()