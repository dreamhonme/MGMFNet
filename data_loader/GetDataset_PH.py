import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PH_dataset(Dataset):
    """
    PH 测试集数据集类
    图像: xxx.npy
    标签: xxx_lesion.npy
    输出格式:
        image: [C, H, W]
        label: [1, H, W]
    """
    def __init__(self, dataset_folder='/PH', with_name=False):
        self.with_name = with_name

        image_dir = os.path.join(dataset_folder, 'image')
        label_dir = os.path.join(dataset_folder, 'label')

        # 自动读取图像
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.folder = [os.path.join(image_dir, f) for f in self.image_list]

        # 自动匹配标签
        self.mask = []
        for f in self.image_list:
            label_name = f.split('.')[0] + '_lesion.npy'
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"标签文件不存在: {label_path}")
            self.mask.append(label_path)

        assert len(self.folder) == len(self.mask), "图像和标签数量不匹配！"

    def __getitem__(self, idx):
        # 读取图像
        image = np.load(self.folder[idx]).astype(np.float32)
        # 如果是 HWC 格式，转换为 CHW
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))  # [H,W,C] -> [C,H,W]
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)   # 灰度图

        # 归一化到 [-1, 1]
        image = (image / 255.0 - 0.5) / 0.5
        image = torch.from_numpy(image)

        # 读取标签
        label = np.load(self.mask[idx]).astype(np.float32)
        if label.ndim == 2:
            label = np.expand_dims(label, axis=0)  # [H,W] -> [1,H,W]
        label = torch.from_numpy(label)

        # 二值化
        label = torch.where(label > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        if self.with_name:
            name = os.path.basename(self.folder[idx])
            return name, image, label
        else:
            return image, label

    def __len__(self):
        return len(self.folder)
