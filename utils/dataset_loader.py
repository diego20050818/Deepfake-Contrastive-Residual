"""
# 数据集加载工具模块

该模块提供了完整的数据集加载、预处理和分割功能，支持单图像分类任务和双图像对比任务。

## 主要组件

### 1. 数据集类
- [DualModelDataLoader] 双图像数据加载器，用于加载成对的真实/伪造图像
- [CustomImageDataset] 单图像数据加载器，用于加载带标签的单张图像

### 2. 数据加载函数

#### 单图像数据加载
- [single_get_dataloader] 创建单图像数据的训练/验证/测试DataLoader
- [single_get_validation_dataloader] 创建单图像验证数据的DataLoader

#### 双图像数据加载
- [pair_get_dataloader] 创建双图像数据的训练/验证/测试DataLoader
- [pair_get_validation_dataloader] 创建双图像验证数据的DataLoader

### 3. 工具类
- [LimitedDataLoader] 限制数据加载器返回的批次数量，用于调试或快速测试

## 使用方法

### 加载单图像数据集

```python
# 基本用法（训练+测试）
train_loader, test_loader = single_get_dataloader(
    dataset_root=Path('data'),
    dataset_names=['dataset1', 'dataset2'],
    batch_size=32,
    transform=transforms_train,
    split=0.8
)

# 包含验证集的分割
train_loader, val_loader, test_loader = single_get_dataloader(
    dataset_root=Path('data'),
    dataset_names=['dataset1', 'dataset2'],
    batch_size=32,
    transform=transforms_train,
    split=0.8,
    validation=True,
    val_ratio_of_remainder=0.5  # 剩余20%中再分割出一半作为验证集
)

# 仅加载验证集
val_loader = single_get_validation_dataloader(
    dataset_root=Path('data'),
    dataset_names=['validation_set'],
    batch_size=32,
    transform=transforms_val
)

# 基本用法（训练+测试）
train_loader, test_loader = pair_get_dataloader(
    dataset_root=Path('data'),
    dataset_names=['dataset1', 'dataset2'],
    batch_size=32,
    transform=transforms_train,
    mode='pair'  # 或 'single'
)

# 包含验证集的分割
train_loader, val_loader, test_loader = pair_get_dataloader(
    dataset_root=Path('data'),
    dataset_names=['dataset1', 'dataset2'],
    batch_size=32,
    transform=transforms_train,
    split=0.8,
    validation=True,
    mode='pair'
)

# 仅加载验证集
val_loader = pair_get_validation_dataloader(
    dataset_root=Path('data'),
    dataset_names=['validation_set'],
    batch_size=32,
    transform=transforms_val,
    mode='pair'
)

# 限制只获取前10个批次的数据
limited_loader = LimitedDataLoader(train_loader, max_batches=10)
for batch_idx, (data, target) in enumerate(limited_loader):
    # 只会执行10次
    pass
"""

import sys
import os
sys.path.append(os.getcwd())

import torch 
from torch.utils.data import DataLoader,Dataset,dataloader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from loguru import logger
from PIL import Image
from pathlib import Path
from typing import Optional,Tuple,List,Literal
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import random

from utils.transforms import transforms_train,transforms_val

# logger.level('INFO')
class DualModelDataLoader:
    def __init__(self,dataset_root:Optional[Path|str],
                 dataset_names:list[str],
                 labels_file_name:str='labels.csv',
                 transform=None,
                 mode:Literal['pair','single']='pair'):
        """双样本数据读取-idx,real_sample,fake_sample

        Args:
            dataset_root (Optional[Path | str]): 数据集根目录
            dataset_names (list[str]): 数据集名称   
            labels_file_name (str, optional): 存储数据集正负样本路径的文件名称.Defaults to 'labels.csv'.
            transform (_type_, optional): 样本变换方式 Defaults to None.
            mode(Literal['pair','single']): 数据集返回方式 可选返回成对正负样本，也可随机返回sample,label单样本形式

        Raises:
            FileNotFoundError: _description_
            FileNotFoundError: _description_
            FileNotFoundError: _description_
            IOError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(dataset_root,Path):
            self.dataset_root = dataset_root
        else:
            self.dataset_root = Path(data_root)

        self.dataset_names = dataset_names
        self.labels_file_name = labels_file_name
        self.transform = transform
        self.mode = mode

        self.real_imgs = list()
        self.fake_imgs = list()

        if not self.dataset_root.exists(): 
            raise FileNotFoundError('dataset root not found:',self.dataset_names)
        
        for dataset_name in tqdm(self.dataset_names,desc='loading dataset'):
            tqdm.write(f"current dataset:{dataset_name}")
            dataset_path = self.dataset_root / dataset_name
            datset_label_file = dataset_path / self.labels_file_name

            if not dataset_path.exists():
                raise FileNotFoundError('dataset not found:',dataset_name)
            if not datset_label_file.is_file():
                raise FileNotFoundError('dataset labels file not found:',datset_label_file)
            
            with open(datset_label_file,'r',encoding='utf-8') as f:
                dataset_info = pd.read_csv(f,header=0)

            real_img_paths = dataset_info['real']
            fake_img_paths = dataset_info['fake']


            self.real_imgs.extend(real_img_paths)
            self.fake_imgs.extend(fake_img_paths)

        if not len(self.real_imgs) == len(self.fake_imgs):
            raise KeyError(f'real imgs len:{len(self.real_imgs)} != fake imgs len:{len(self.fake_imgs)}')

        self.pair_data = zip(self.real_imgs,self.fake_imgs)

        self.total_samples = len(self.real_imgs)
        logger.info(f'there are {self.total_samples} pair samples')

    def __len__(self):
        """
        获取数据集的总样本数

        Returns:
            int: 数据集的总样本数
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        根据索引获取样本数据

        Args:
            idx (int): 索引值

        Returns:
            Tuple[Tensor, Tensor] 或 Tuple[Tensor, int]:
                - 如果 mode 为 'pair'，返回 (real_tensor, fake_tensor)
                - 如果 mode 为 'single'，返回 (img_tensor, label)
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.total_samples}")

        real_img_path = self.real_imgs[idx]
        fake_img_path = self.fake_imgs[idx]

        try:
            real_img = Image.open(real_img_path).convert('RGB')
            fake_img = Image.open(fake_img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load images at index {idx}: {e}")
            raise IOError(f"Error loading images at index {idx}")

        if self.transform:
            real_tensor = self.transform(real_img)
            fake_tensor = self.transform(fake_img)
        else:
            real_tensor = real_img
            fake_tensor = fake_img

        if self.mode == 'pair':
            return real_tensor, fake_tensor
        elif self.mode == 'single':
            # 随机选择一个样本和对应的标签
            if random.random() > 0.5:
                return real_tensor, 1  # 真实样本，标签为 1
            else:
                return fake_tensor, 0  # 伪造样本，标签为 0
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Supported modes are 'pair' and 'single'.")


class CustomImageDataset(Dataset):
    def __init__(self, dataset_root:Optional[Path|str],
                 dataset_names:list[str],
                 labels_file_name:str='labels.csv', 
                 transform=None):
        """单样本数据读取-sample,label 

        Args:
            dataset_root (Path): 数据集根目录
            dataset_names (list[str]): 数据集名称
            labels_file_name (str, optional): 存储数据集路(path)和标签字段(label)的文件 Defaults to 'labels.csv'.
            transform (_type_, optional): 数据集变换策略. Defaults to None.

        Raises:
            FileNotFoundError
        """
        if isinstance(dataset_root,Path):
            self.dataset_root = dataset_root
        else:
            self.dataset_root = Path(data_root)

        self.dataset_names = dataset_names
        self.labels_file_name = labels_file_name
        self.transform = transform

        if not self.dataset_root.exists(): 
            raise FileNotFoundError('dataset root not found:',self.dataset_names)
        
        self.all_img_paths = list()
        self.all_img_labels = list()
        
        for dataset_name in tqdm(self.dataset_names,desc='loading dataset'):
            tqdm.write(f"current dataset:{dataset_name}")
            dataset_path = self.dataset_root / dataset_name
            datset_label_file = dataset_path / self.labels_file_name

            if not dataset_path.exists():
                raise FileNotFoundError('dataset not found:',dataset_name)
            if not datset_label_file.is_file():
                raise FileNotFoundError('dataset labels file not found:',datset_label_file)
            
            with open(datset_label_file,'r',encoding='utf-8') as f:
                dataset_info = pd.read_csv(f,header=0)

            img_paths = dataset_info['path']
            img_labels = dataset_info['label'].astype(np.int16)


            self.all_img_paths.extend(img_paths)
            self.all_img_labels.extend(img_labels)

        # 统计总样本数和负样本数
        total_samples = len(self.all_img_labels)
        negative_count = self.all_img_labels.count(0)

        # 计算占比
        negative_ratio = (negative_count / total_samples) * 100 if total_samples > 0 else 0.0
        positive_ratio = 100.0 - negative_ratio  # 正样本占比 = 100% - 负样本占比

        logger.info(
            f"there are {negative_ratio:.2f}% negative samples and {positive_ratio:.2f}% positive samples"
        )
        self.all_img_labels = torch.tensor(self.all_img_labels)
        
        logger.info(f"successfully loaded {len(self.all_img_paths)} samples (*^▽^*)")


            
    def __len__(self):
        """获取数据集长度

        Returns:
            _type_: _description_
        """
        return len(self.all_img_paths)
        pass
    
    def __getitem__(self, idx:int):
        """指定索引访问数据的方法

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        img_path:Optional[str|Path] = self.all_img_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB') # type:ignore
        except Exception as e:
            logger.error("failed to covert img to RGB:",e)
            raise IOError(f"{img_path}")
        
        label = self.all_img_labels[idx]
        
        if self.transform:
            image = self.transform(image)

            
        # return image, label
@logger.catch()
def single_get_dataloader(
    dataset_root: Path,
    dataset_names: List[str],
    batch_size: int,
    transform: transforms.Compose,
    split: float = 0.8, # 训练集占总数据集的比例 (Train: 0.8, Test: 0.2)
    validation: bool = False, # 是否进行三方分割 (Train/Val/Test)
    val_ratio_of_remainder: float = 0.5, # 如果 validation=True, 剩余部分中分给验证集的比例
    random_seed: int = 42,
    num_works: int = 4,
    labels_file_name: str = 'labels.csv'
) -> Tuple[DataLoader, ...]:
    """
    创建完整数据集并分割为训练集、(验证集) 和测试集，返回对应的 DataLoader。

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        split (float, optional): 训练集占总数据集的比例. Defaults to 0.8.
        validation (bool, optional): 是否进行三方分割 (Train/Val/Test). Defaults to False.
        val_ratio_of_remainder (float, optional): 如果 validation=True, 剩余部分中分给验证集的比例. Defaults to 0.5.
        random_seed (int, optional): 随机种子（保证分割可复现）. Defaults to 42.
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.

    Returns:
        Tuple[DataLoader, ...]: 
            如果 validation=False: (train_dataloader, test_dataloader)
            如果 validation=True: (train_dataloader, val_dataloader, test_dataloader)
    """
    # 验证分割比例有效性
    if not (0 < split < 1):
        raise ValueError(f"split 必须在 (0, 1) 范围内，当前值: {split}")
    if validation and not (0 < val_ratio_of_remainder < 1):
        raise ValueError(f"val_ratio_of_remainder 必须在 (0, 1) 范围内，当前值: {val_ratio_of_remainder}")
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = CustomImageDataset(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform
    )
    dataset_size = len(full_dataset)
    logger.info(f"完整数据集大小: {dataset_size}")
    
    # 设置随机种子保证可复现性
    random.seed(random_seed)
    torch.manual_seed(random_seed) 
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # 2. 数据集分割
    
    # A. 训练集大小
    train_size = int(dataset_size * split)
    train_indices = indices[:train_size]
    
    # B. 剩余部分
    remainder_indices = indices[train_size:]
    remainder_size = len(remainder_indices)
    
    val_dataset = None
    val_dataloader = None
    
    if validation:
        # 三方分割：Train / Val / Test
        
        # 剩余部分按 val_ratio_of_remainder 分割给 Val 和 Test
        val_size = int(remainder_size * val_ratio_of_remainder)
        
        val_indices = remainder_indices[:val_size]
        test_indices = remainder_indices[val_size:]
        
        # 创建验证集
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}, 测试集大小: {len(test_indices)}")
    else:
        # 二方分割：Train / Test
        test_indices = remainder_indices # 剩余部分全部作为测试集
        logger.info(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

    # 创建训练集和测试集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 3. 定义 collate_fn 处理损坏文件
    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            # 返回 None, None 让上层调用者可以跳过这个批次
            return None, None 
        return default_collate(batch)
    
    # 4. 构建 DataLoader
    
    # 训练集 DataLoader（打乱）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 验证集 DataLoader（不打乱）
    if validation:
        val_dataloader = DataLoader(
            val_dataset, # type:ignore
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_works,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # 测试集 DataLoader（不打乱）
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    end_time = time.time()
    logger.info(f"数据集加载与分割完成，总耗时: {end_time - start_time:.2f}s")
    
    if validation:
        return (train_dataloader, val_dataloader, test_dataloader) # type:ignore
    else:
        return (train_dataloader, test_dataloader)

@logger.catch()
def single_get_validation_dataloader(
        dataset_root: Path,
        dataset_names: List[str],
        batch_size: int,
        transform: transforms.Compose,
        num_works: int = 4,
        labels_file_name: str = 'labels.csv'
) -> DataLoader:
    """
    创建完整数据集作为验证集，返回对应的 DataLoader

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.

    Returns:
        DataLoader: 验证集 DataLoader
    """
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = CustomImageDataset(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform
    )
    logger.info(f"完整验证集大小: {len(full_dataset)}")
    

    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        return default_collate(batch)
    

    val_dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    
    end_time = time.time()
    logger.info(f"验证集加载完成，总耗时: {end_time - start_time:.2f}s")
    
    return val_dataloader

from itertools import islice

@logger.catch()
def pair_get_dataloader(
    dataset_root: Path,
    dataset_names: List[str],
    batch_size: int,
    transform: transforms.Compose,
    split: float = 0.8,  # 训练集占总数据集的比例 (Train: 0.8, Test: 0.2)
    validation: bool = False,  # 是否进行三方分割 (Train/Val/Test)
    val_ratio_of_remainder: float = 0.5,  # 如果 validation=True, 剩余部分中分给验证集的比例
    random_seed: int = 42,
    num_works: int = 4,
    labels_file_name: str = 'labels.csv',
    mode: Literal['pair', 'single'] = 'pair'
) -> Tuple[DataLoader, ...]:
    """
    创建完整双模型数据集并分割为训练集、(验证集) 和测试集，返回对应的 DataLoader。

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        split (float, optional): 训练集占总数据集的比例. Defaults to 0.8.
        validation (bool, optional): 是否进行三方分割 (Train/Val/Test). Defaults to False.
        val_ratio_of_remainder (float, optional): 如果 validation=True, 剩余部分中分给验证集的比例. Defaults to 0.5.
        random_seed (int, optional): 随机种子（保证分割可复现）. Defaults to 42.
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.
        mode (Literal['pair', 'single'], optional): 数据加载模式. Defaults to 'pair'.

    Returns:
        Tuple[DataLoader, ...]: 
            如果 validation=False: (train_dataloader, test_dataloader)
            如果 validation=True: (train_dataloader, val_dataloader, test_dataloader)
    """
    # 验证分割比例有效性
    if not (0 < split < 1):
        raise ValueError(f"split 必须在 (0, 1) 范围内，当前值: {split}")
    if validation and not (0 < val_ratio_of_remainder < 1):
        raise ValueError(f"val_ratio_of_remainder 必须在 (0, 1) 范围内，当前值: {val_ratio_of_remainder}")
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = DualModelDataLoader(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform,
        mode=mode
    )
    dataset_size = len(full_dataset)
    logger.info(f"完整数据集大小: {dataset_size}")
    
    # 设置随机种子保证可复现性
    random.seed(random_seed)
    torch.manual_seed(random_seed) 
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # 2. 数据集分割
    
    # A. 训练集大小
    train_size = int(dataset_size * split)
    train_indices = indices[:train_size]
    
    # B. 剩余部分
    remainder_indices = indices[train_size:]
    remainder_size = len(remainder_indices)
    
    val_dataset = None
    val_dataloader = None
    
    if validation:
        # 三方分割：Train / Val / Test
        
        # 剩余部分按 val_ratio_of_remainder 分割给 Val 和 Test
        val_size = int(remainder_size * val_ratio_of_remainder)
        
        val_indices = remainder_indices[:val_size]
        test_indices = remainder_indices[val_size:]
        
        # 创建验证集
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}, 测试集大小: {len(test_indices)}")
    else:
        # 二方分割：Train / Test
        test_indices = remainder_indices  # 剩余部分全部作为测试集
        logger.info(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

    # 创建训练集和测试集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 3. 定义 collate_fn 处理损坏文件
    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            # 返回 None, None 让上层调用者可以跳过这个批次
            return None, None 
        return default_collate(batch)
    
    # 4. 构建 DataLoader
    
    # 训练集 DataLoader（打乱）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 验证集 DataLoader（不打乱）
    if validation:
        val_dataloader = DataLoader(
            val_dataset,  # type:ignore
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_works,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # 测试集 DataLoader（不打乱）
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    end_time = time.time()
    logger.info(f"数据集加载与分割完成，总耗时: {end_time - start_time:.2f}s")
    
    if validation:
        return (train_dataloader, val_dataloader, test_dataloader)  # type:ignore
    else:
        return (train_dataloader, test_dataloader)


@logger.catch()
def pair_get_validation_dataloader(
        dataset_root: Path,
        dataset_names: List[str],
        batch_size: int,
        transform: transforms.Compose,
        num_works: int = 4,
        labels_file_name: str = 'labels.csv',
        mode: Literal['pair', 'single'] = 'pair'
) -> DataLoader:
    """
    创建完整双模型数据集作为验证集，返回对应的 DataLoader

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.
        mode (Literal['pair', 'single'], optional): 数据加载模式. Defaults to 'pair'.

    Returns:
        DataLoader: 验证集 DataLoader
    """
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = DualModelDataLoader(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform,
        mode=mode
    )
    logger.info(f"完整验证集大小: {len(full_dataset)}")
    

    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        return default_collate(batch)
    

    val_dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    
    end_time = time.time()
    logger.info(f"验证集加载完成，总耗时: {end_time - start_time:.2f}s")
    
    return val_dataloader

class LimitedDataLoader:
    def __init__(self, dataloader, max_batches):
        """限制数据集 对数据集进行切片

        Args:
            dataloader (dataloader): 数据集实例
            max_batches (int): 前n个批次
        """
        self.dataloader = dataloader
        self.max_batches = max_batches
    
    def __iter__(self):
        return islice(self.dataloader, self.max_batches)
    
    def __len__(self):
        return min(self.max_batches, len(self.dataloader))


if __name__ == '__main__':
    data_root = Path('dataset')
    data_name = ['dataset/Celeb-DF-pair']

    train_loader, val_loader, test_loader = pair_get_dataloader(
        dataset_root=data_root,
        dataset_names=data_name,
        batch_size=32,
        transform=transforms_train,
        split=0.8,
        validation=True,
        mode='pair'
    )

    logger.debug(train_loader)










