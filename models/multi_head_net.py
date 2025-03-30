from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module


class MultiHeadNet(Module):
    def __init__(self, in_channels: int =1, num_classes: int =10) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),     # [N, 64, 28, 28]
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),     # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),     # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # [N, 256, 1, 1]
            nn.Flatten(),   # [N, 256]
            nn.Linear(256, num_classes) # [N, 10]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),   # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),   # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),    # [N, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes + 1, 3, stride=1, padding=1, bias=True)   # [N, 1, 28, 28]
        )

        # キーポイントマップは値が0～1なので最後はシグモイド
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, self.num_classes, 1, stride=1, padding=0),  # [N, 10, 7, 7]
            nn.Sigmoid()
        )

        # オフセットマップは値が正負をとるので最後は恒等写像（Conv2dのまま）
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 2, 1, stride=1, padding=0),  # [N, 2, 7, 7]
        )
        
        # サイズマップは値が0～なので最後はReLU
        self.size_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 2, 1, stride=1, padding=0),   # [N, 2, 7, 7]
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """順伝播を行う

        Args:
            x (Tensor): 入力。Tensor形状は[N, in_channels, H, W]

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            キーポイントマップ、オフセットマップ、サイズマップ。
            Tensor形状は[N, num_classes, H/4, W/4]、[N, 2, H/4, W/4]、[N, 2, H/4, W/4]
        """
        h = self.encoder(x)

        # return self.keypoint_head(h), self.offset_head(h), self.size_head(h)
        outputs = {
            "classification_logits": self.classification_head(h),
            "segmentation_logits": self.decoder(h),
            "detection_keypoints": self.keypoint_head(h),
            "detection_offsets": self.offset_head(h),
            "detection_sizes": self.size_head(h)
        }
        return outputs


if __name__ == '__main__':
    """テストコード
    """
    batch_size = 64
    in_channel = 1
    num_classes = 10
    image_height = 128
    image_width = 128
    
    images = torch.rand(batch_size, in_channel, image_height, image_width)
    print(f'images: {images.shape}')

    model = MultiHeadNet(in_channel, num_classes)

    outputs = model(images)
    classification_logits = outputs["classification_logits"]
    segmentation_logits = outputs["segmentation_logits"]
    detection_keypoints = outputs["detection_keypoints"]
    detection_offsets = outputs["detection_offsets"]
    detection_sizes = outputs["detection_sizes"]

    # print(f'keypoints: {keypoints.shape}  offsets: {offsets.shape}  sizes: {sizes.shape}')
    print(f'classification logits: {classification_logits.shape}')
    print(f'segmentation logits: {segmentation_logits.shape}')
    print(f'detection keypoints: {detection_keypoints.shape}')
    print(f'detection offsets: {detection_offsets.shape}')
    print(f'detection sizes: {detection_sizes.shape}')