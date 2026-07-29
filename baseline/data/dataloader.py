from torch.utils.data import DataLoader
from dataset import PickleballDataset
from collate import collate_fn


def build_dataloader(samples, batch_size=4):
    dataset = PickleballDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    return loader