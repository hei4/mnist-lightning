import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import albumentations as A

class TripletMnist(Dataset):
    def __init__(self, root, train, download, threshold, sigma, transform=None):
        super().__init__()
        self.mnist = MNIST(root, train, download=download)
        self.threshold = threshold
        self.sigma = sigma
        self.transform = transform
        self.grid_y, self.grid_x = torch.meshgrid(
            torch.arange(7),
            torch.arange(7),
            indexing='ij'
        )    # ガウス分布に使用する座標
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
        image, label = self.mnist[index]

        image = np.array(image, dtype=np.float32) / 255.
        mask = np.where(image > self.threshold, label + 1, 0)   # 閾値を超えた画素にラベル+1の値

        binary_mask = np.where(image > self.threshold, True, False)    # 閾値を超えた画素がTrueのマスク

        y_indices, x_indices = np.where(binary_mask)   # Trueになる画素の座標

        x_min = x_indices.min().astype(np.float32)
        y_min = y_indices.min().astype(np.float32)
        x_max = x_indices.max().astype(np.float32) + 1.     # ピクセルを含むように1加算
        y_max = y_indices.max().astype(np.float32) + 1.

        bbox = [x_min, y_min, x_max, y_max]

        bboxes = []     # 不定個のバウンディングボックスのリスト
        bbox_labels = []     # 不定個のラベルのリスト

        bboxes.append(bbox)     # 本来は物体の個数は不定だが、MNISTでは個数1で固定
        bbox_labels.append(label)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask, bboxes=bboxes, labels=bbox_labels)
            image = transformed['image']
            mask = transformed['mask'].to(torch.int64)
            bboxes = transformed['bboxes']
        
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bbox_labels = torch.tensor(bbox_labels, dtype=torch.long)

        bbox_keypoint, bbox_offset, bbox_size = self.__make_maps(bboxes, bbox_labels)    # マップの作成
        
        return image, label, mask, bboxes, bbox_labels, bbox_keypoint, bbox_offset, bbox_size
    
    def __make_maps(self, bboxes, bbox_labels):
        """バウンディングボックスとラベルからマップを生成する

        Args:
            bboxes (Tensor): バウンディングボックス。Tensor形状は[M, 4]
            labels (Tensor): ラベル。Tensor形状は[M]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
            キーポイントマップ、オフセットマップ、サイズマップ
            キーポイントマップのTensor形状は[10, 7, 7]
            オフセットマップのTensor形状は[2, 7, 7]
            サイズマップのTensor形状は[2, 7, 7]
        """
        keypoint = torch.zeros([10, 7, 7], dtype=torch.float32)     # 1/4縮小のマップ
        offset = torch.zeros([2, 7, 7], dtype=torch.float32)        # xとyで2チャネル
        size = torch.zeros([2, 7, 7], dtype=torch.float32)          # 高さと幅で2チャネル

        # バウンディングボックスが複数あれば以下のループでボックス個数だけマップに加算する
        # 当然、MNISTでは以下のループは1回しか実行されないので注意
        for bbox, label in zip(bboxes, bbox_labels):
            x_center = (bbox[0] + bbox[2]) / 2.
            y_center = (bbox[1] + bbox[3]) / 2.
            W_index = int(x_center / 4.)    # x中心に対応するインデックス
            H_index = int(y_center / 4.)    # y中心に対応するインデックス

            # 中心位置が1のガウシアンをキーポイントマップに加算する
            keypoint[label] += torch.exp(-((self.grid_x - W_index) **
                                         2 + (self.grid_y - H_index)**2) / (2 * self.sigma**2))

            # オフセットの設定。キーポイントマップでのピクセル中心は元画像の2ピクセルずれた位置
            offset[0, H_index, W_index] += x_center - \
                4. * (W_index + 0.5)  # x offset
            offset[1, H_index, W_index] += y_center - \
                4. * (H_index + 0.5)  # y offset

            size[0, H_index, W_index] += bbox[2] - bbox[0]  # width
            size[1, H_index, W_index] += bbox[3] - bbox[1]  # height

        # 複数回の加算を考慮して、上限1下限0にスライスする
        keypoint = torch.clamp(keypoint, min=0., max=1.)    
        
        return keypoint, offset, size

if __name__ == "__main__":
    from albumentations.pytorch import transforms as AT
    root = "/home/sika/Datasets"
    threshold = 0.5
    sigma = 1.
    transform = A.Compose(
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
    
    dataset = TripletMnist(root, True, True, threshold, sigma, transform)

    image, label, mask, bboxes, bbox_labels, bbox_keypoint, bbox_offset, bbox_size = dataset[0]
    print(image.shape)
    print(label)
    print(mask.shape)
    print(bboxes.shape)
    print(bbox_labels.shape)
    print(bbox_keypoint.shape)
    print(bbox_offset.shape)
    print(bbox_size.shape)