import lightning as L

from models.multi_task_lightning_module import MultiTaskLightningModule
from data.multi_task_mnist_module import MutlTaskMnistModule

def main():
    # init the autoencoder
    model_module = MultiTaskLightningModule.load_from_checkpoint("lightning_logs/version_18/checkpoints/epoch=9-step=1000.ckpt")

    # disable randomness, dropout, etc...
    model_module.eval()

    # setup data module
    root = "/home/sika/Datasets"
    threshold = 0.5
    sigma = 1.
    data_module = MutlTaskMnistModule(root, train=False, thresold=threshold, sigma=sigma)

    # initialize the Trainer
    trainer = L.Trainer()

    # test the model
    trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()