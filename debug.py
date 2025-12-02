from utils.dataset_loader import pair_get_dataloader
from utils.transforms import transforms_train
from loguru import logger
from pathlib import Path

data_root = Path('dataset')
data_name = ['Celeb-DF-pair']

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
