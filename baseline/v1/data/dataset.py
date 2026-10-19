import torch
import numpy as np
from torch.utils.data import Dataset


class PickleballDataset(Dataset):
    def __init__(self, samples, num_frames = 60 )