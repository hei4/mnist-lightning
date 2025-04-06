import os
import torch
from torch.nn import functional as F
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torchmetrics import Accuracy, Dice
from torchmetrics.detection import MeanAveragePrecision

from multi_head_net import MultiHeadNet
from criteria.pointnet_loss import PointNetLoss
from utils.util import make_bboxes_list

# define the LightningModuleP
class MultiTaskLightningModule(L.LightningModule):
    def __init__(self, train=False):
        super().__init__()

        # モデルサマリーで使用する入力形状
        self.example_input_array = torch.zeros(100, 1, 28, 28)

        # ハイパーパラメータの保存
        self.save_hyperparameters()

        self.classification_criterion = nn.CrossEntropyLoss()
        self.segmentation_criterion = nn.CrossEntropyLoss()
        self.detection_criterion = PointNetLoss()

        self.train_classification_metric = Accuracy(task="multiclass", num_classes=10)
        self.val_classification_metric = self.train_classification_metric.clone()
        self.test_classification_metric = self.train_classification_metric.clone()

        self.train_segmentation_metric = Dice(num_classes=10+1, multiclass=True)
        self.val_segmentation_metric = self.train_segmentation_metric.clone()
        self.test_segmentation_metric = self.train_segmentation_metric.clone()

        self.train_detection_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.val_detection_metric = self.train_detection_metric.clone()
        self.test_detection_metric = self.train_detection_metric.clone()

        if train == True:
            self.model = MultiHeadNet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, labels, masks, target_bboxes_list, target_labels_list, target_keypoints, target_offsets, target_sizes = batch
        outputs = self.model(images)
        classification_logits = outputs["classification_logits"]
        segmentation_logits = outputs["segmentation_logits"]
        detection_keypoints = outputs["detection_keypoints"]
        detection_offsets = outputs["detection_offsets"]
        detection_sizes = outputs["detection_sizes"]
        
        classification_loss = self.classification_criterion(classification_logits, labels)
        self.log("train/classification/loss", classification_loss)
        self.train_classification_metric(classification_logits, labels)
        self.log('train/classification/Accuracy', self.train_classification_metric, on_step=False, on_epoch=True)

        segmentation_loss = self.segmentation_criterion(segmentation_logits, masks)
        self.log("train/segmentation/loss", segmentation_loss)
        self.train_segmentation_metric(segmentation_logits, masks)
        self.log("train/segmentation/Dice", self.train_segmentation_metric, on_step=False, on_epoch=True)

        detection_loss = self.detection_criterion(
            detection_keypoints, detection_offsets, detection_sizes,
            target_keypoints, target_offsets, target_sizes
        )
        self.log("train/detection/loss", detection_loss)

        # mAPの計算は時間がかかるのでtraining_stepでは割愛する

        loss = classification_loss + segmentation_loss + detection_loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, labels, masks, target_bboxes_list, target_labels_list, target_keypoints, target_offsets, target_sizes = batch
        outputs = self.model(images)
        classification_logits = outputs["classification_logits"]
        segmentation_logits = outputs["segmentation_logits"]
        detection_keypoints = outputs["detection_keypoints"]
        detection_offsets = outputs["detection_offsets"]
        detection_sizes = outputs["detection_sizes"]

        classification_loss = self.classification_criterion(classification_logits, labels)
        self.log("val/classification/loss", classification_loss)
        self.val_classification_metric(classification_logits, labels)
        self.log('val/classification/Accuracy', self.val_classification_metric, on_step=False, on_epoch=True)

        segmentation_loss = self.segmentation_criterion(segmentation_logits, masks)
        self.log("val/segmentation/loss", segmentation_loss)
        self.val_segmentation_metric(segmentation_logits, masks)
        self.log("val/segmentation/Dice", self.val_classification_metric, on_step=False, on_epoch=True)

        detection_loss = self.detection_criterion(
            detection_keypoints, detection_offsets, detection_sizes,
            target_keypoints, target_offsets, target_sizes
        )
        self.log("val/detection/loss", detection_loss)

        pred_bboxes_list, pred_scores_list, pred_labels_list = make_bboxes_list(
            detection_keypoints, detection_offsets, detection_offsets, k=3, device=self.device)     # 3個までのボックスを推定

        pred_dict_list = []
        for pred_bboxes, pred_scores, pred_labels in zip(pred_bboxes_list, pred_scores_list, pred_labels_list):
            pred_dict_list.append({
                "boxes": pred_bboxes,
                "scores": pred_scores,
                "labels": pred_labels
            })
        
        target_dict_list = []
        for target_bboxes, target_labels in zip(target_bboxes_list, target_labels_list):
            target_dict_list.append({
                "boxes": target_bboxes,
                "labels": target_labels
            })
        
        map_dict = self.val_detection_metric(pred_dict_list, target_dict_list)
        self.log("val/detection/mAP", map_dict["map"], on_step=False, on_epoch=True)

        loss = classification_loss + segmentation_loss + detection_loss
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, labels, masks, target_bboxes_list, target_labels_list, target_keypoints, target_offsets, target_sizes = batch
        outputs = self.model(images)
        classification_logits = outputs["classification_logits"]
        segmentation_logits = outputs["segmentation_logits"]
        detection_keypoints = outputs["detection_keypoints"]
        detection_offsets = outputs["detection_offsets"]
        detection_sizes = outputs["detection_sizes"]

        classification_loss = self.classification_criterion(classification_logits, labels)
        self.log("test/classification/loss", classification_loss)
        self.test_classification_metric(classification_logits, labels)
        self.log('test/classification/acc', self.test_classification_metric, on_step=False, on_epoch=True)

        segmentation_loss = self.segmentation_criterion(segmentation_logits, masks)
        self.log("test/segmentation/loss", segmentation_loss)
        self.test_segmentation_metric(segmentation_logits, masks)
        self.log("test/segmentation/dice", self.test_segmentation_metric, on_step=False, on_epoch=True)

        detection_loss = self.detection_criterion(
            detection_keypoints, detection_offsets, detection_sizes,
            target_keypoints, target_offsets, target_sizes
        )
        self.log("test/detection/loss", detection_loss)

        pred_bboxes_list, pred_scores_list, pred_labels_list = make_bboxes_list(
            detection_keypoints, detection_offsets, detection_offsets, k=3, device=self.device)     # 3個までのボックスを推定
        
        pred_dict_list = []
        for pred_bboxes, pred_scores, pred_labels in zip(pred_bboxes_list, pred_scores_list, pred_labels_list):
            pred_dict_list.append({
                "boxes": pred_bboxes,
                "scores": pred_scores,
                "labels": pred_labels
            })
        
        target_dict_list = []
        for target_bboxes, target_labels in zip(target_bboxes_list, target_labels_list):
            target_dict_list.append({
                "boxes": target_bboxes,
                "labels": target_labels
            })
        
        map_dict = self.test_detection_metric(pred_dict_list, target_dict_list)
        self.log("test/detection/mAP", map_dict["map"], on_step=False, on_epoch=True)

        loss = classification_loss + segmentation_loss + detection_loss
        self.log("test/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model_module = MultiTaskLightningModule(train=True)
    print(model_module)

    model_module = MultiTaskLightningModule(train=False)
    print(model_module)
