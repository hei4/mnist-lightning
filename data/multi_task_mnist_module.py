from copy import deepcopy
import torch
import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms as VT
import albumentations as A
from albumentations.pytorch import transforms as AT

from triplet_mnist import TripletMnist

# 画像やマップは1つのTensorに結合し、ボックスとボックスラベルはリストのままにする関数
def collate_fn(batch):
    images, labels, masks, bboxes, bbox_labels, bbox_keypoints, bbox_offsets, bbox_sizes = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels), torch.stack(masks, dim=0), \
        bboxes, bbox_labels, torch.stack(bbox_keypoints, dim=0), torch.stack(bbox_offsets, dim=0), torch.stack(bbox_sizes, dim=0)

class MutlTaskMnistModule(L.LightningDataModule):
    def __init__(self, root, train=False, thresold=0.5, sigma=1., seed=None):
        super().__init__()
        self.root = root
        
        self.eval_transform = A.Compose(
            [AT.ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

        if train == True:
            self.train_transform = A.Compose(
                [
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_px=(-2, 2),
                        rotate=(-10, 10),
                        shear=(-10, 10),
                        always_apply=True
                    ),
                    AT.ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
            )

        self.threshold = thresold
        self.sigma = sigma
        self.seed = seed

        self.loader_kwargs = {
            "batch_size": 500,
            "num_workers": 4,
            "pin_memory": True,
            "collate_fn": collate_fn
        }

    def prepare_data(self):
        # ダウンロードなどCPU上の1つのプロセスで実行する処理
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_val_set = TripletMnist(self.root, True, False, self.threshold, self.sigma, transform=self.train_transform)
            self.train_set, self.val_set = random_split(
                train_val_set,
                [50000, 10000],
                generator=torch.Generator().manual_seed(self.seed)
            )
            self.val_set = deepcopy(self.val_set)
            self.val_set.dataset.transform = self.eval_transform

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = TripletMnist(self.root, False, False, self.threshold, self.sigma, transform=self.eval_transform)

        if stage == "predict":
            self.predict_set = TripletMnist(self.root, False, False, self.threshold, self.sigma, transform=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, drop_last=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, drop_last=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, drop_last=False, **self.loader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, shuffle=False, drop_last=False, **self.loader_kwargs)

if __name__ == "__main__":
    root = "/home/sika/Datasets"
    threshold = 0.5
    sigma = 1.
    seed = 0
    data_module = MutlTaskMnistModule(root, True, threshold, sigma, seed)

    data_module.setup(stage="fit")
    print("training set")
    print(len(data_module.train_set))
    print(data_module.train_set.dataset.transform)

    print("validation set")
    print(len(data_module.val_set))
    print(data_module.val_set.dataset.transform)

    data_module.setup(stage="test")
    print(len(data_module.test_set))
    print(data_module.test_set.transform)

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels, masks, bboxes_list, bbox_labels_list, bbox_keypoints, bbox_offsets, bbox_sizes = batch
    print(images.shape)
    print(labels.shape)
    print(masks.shape)
    print(type(bboxes_list), len(bboxes_list), bboxes_list[0].shape)
    print(type(bbox_labels_list), len(bbox_labels_list), bbox_labels_list[0].shape)
    print(bbox_keypoints.shape)
    print(bbox_offsets.shape)
    print(bbox_sizes.shape)

