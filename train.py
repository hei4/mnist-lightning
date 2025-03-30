import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from models.multi_task_lightning_module import MultiTaskLightningModule
from data.multi_task_mnist_module import MutlTaskMnistModule

def main():
    # setup random seed
    seed = 0
    L.seed_everything(seed, workers=True)

    # init the model module
    model_module = MultiTaskLightningModule(train=True)

    # setup datamodule
    root = "/home/sika/Datasets"
    threshold = 0.5
    sigma = 1.
    data_module = MutlTaskMnistModule(root, train=True, thresold=threshold, sigma=sigma, seed=seed)

    # setup logger
    mlflow_logger = MLFlowLogger(experiment_name="lightning_logs")

    # setup Traniner
    trainer = L.Trainer(max_epochs=10, logger=mlflow_logger)
    
    # fitting
    trainer.fit(model=model_module, datamodule=data_module)

if __name__ == "__main__":
    main()