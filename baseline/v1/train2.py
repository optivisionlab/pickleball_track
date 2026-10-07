import cv2
from data.dataset import PickleballDataset
from torch.utils.data import DataLoader
from data.collate import collate_fn
from models import AVModel
import random
import numpy as np
import torch


def seed_random(seed = 42):
    random.seed(seed) #seed for python
    np.random.seed(seed) #seed for numpy
    torch.manual_seed_all(seed) #seed for torch
    torch.cuda.manual_seed_all(seed) #seed for CUDA
    torch.backends.cudnn.deterministic = False #off algorithms deterministic 
    torch.backends.cudnn.benchmark = True #on best algorithms 
    
def load_dataloader(samples):
    train_dataset = PickleballDataset(
        samples,
        target_frames = 60,
        image_sz=IMAGE_SIZE,
        train = True
    )
    val_dataset = PickleballDataset(
        samples,
        target_frames = 60,
        image_sz = IMAGE_SIZE,
        train = False,
        num_workers = 8,
        pin_memory = True
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 8
    )
    