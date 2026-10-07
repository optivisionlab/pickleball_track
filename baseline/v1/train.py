import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEModel
from data.build_labels import build_index       
from data.dataset import PickleballDataset
from data.collate import collate_fn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from models.model import AVModel
from configs.config import *
import json


os.makedirs(SAVE_DIR, exist_ok = True)
logging.basicConfig(
    filename=os.path.join(SAVE_DIR,"train.log"),
    level=logging.INFO,
)
logger = logging.getLogger()


def seed_everything(seed = 42): 
    random.seed(seed) #cố định seed cho python
    np.random.seed(seed)  # cố định seed cho numpy
    torch.manual_seed(seed) # cố định seed cho torch
    torch.cuda.manual_seed_all(seed)  # cố định seed cho CUDA
    torch.backends.cudnn.deterministic = False  #không sử dụng thuật toán deterministic để đạt tốc độ cao hơn
    torch.backends.cudnn.benchmark = True   #benchmark để tìm ra thuật toán tính toán nhanh nhất trên CuDA


def build_dataloader(train_samples, val_samples):
    train_dataset = PickleballDataset(
        train_samples,
        target_frames=NUM_FRAMES,
        image_sz=IMAGE_SIZE,
        train=True,
    )

    val_dataset = PickleballDataset(
        val_samples,
        target_frames=NUM_FRAMES,
        image_sz=IMAGE_SIZE,
        train=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn
    )
    return train_loader,val_loader

def load_videomae(): 
    return VideoMAEModel.from_pretrained(VIDEO_MODEL_NAME)

def load_beats():
    import sys
    base_unilm = "/data/Cuong/pickleball/pickleball_track/baseline/v1/unilm"
    beats_dir = os.path.join(base_unilm, "beats")
    sys.path.append(base_unilm)
    sys.path.append(beats_dir)
    from beats.BEATs import BEATs, BEATsConfig
    checkpoint = torch.load(BEATS_CHECKPOINT, map_location = "cpu")
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def build_model(): 
    return AVModel(load_videomae(),load_beats(),fusion_dim = 768)

def freeze_backbone(model):
    for p in model.video_encoder.parameters(): 
        p.requires_grad = False
    for p in model.audio_encoder.parameters(): 
        p.requires_grad = False

def unfreeze_and_update_optimizer(model, optimizer, epoch):
    unfreeze_last_blocks(model)
    new_lr = LR * 0.1
    new_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=new_lr,
        weight_decay=WEIGHT_DECAY
    )
    return new_optimizer

def unfreeze_last_blocks(model):
    video_blocks = getattr(model.video_encoder.model.encoder,"layer",None)
    if video_blocks is not None:
        for block in video_blocks[-2:]:
            for p in block.parameters(): 
                p.requires_grad = True
    if hasattr(model.audio_encoder.model,"encoder"):
        audio_blocks = model.audio_encoder.model.encoder.layers
        for block in audio_blocks[-2:]:
            for p in block.parameters(): 
                p.requires_grad = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    preds, labels = [], []

    for batch in tqdm(loader):
        outputs = {}
        for k,v in batch.items():
            outputs[k] = v.to(device,non_blocking = True) if isinstance(v,torch.Tensor) else v
        optimizer.zero_grad(set_to_none=True)
        predict = model(outputs)
        loss = criterion(predict, outputs["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds.extend(predict.argmax(1).detach().cpu().numpy())
        labels.extend(outputs["label"].cpu().numpy())

    return {
        "loss": total_loss / len(loader),
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

@torch.no_grad()
def validate(model,loader,criterion,device):
    model.eval()
    total_loss = 0.0 
    preds = [] 
    labels = [] 
    pbar = tqdm(loader)
    for batch in pbar:
        outputs = {}
        for k,v in batch.items():
            outputs[k] = v.to(device,non_blocking = True) if isinstance(v,torch.Tensor) else v
        predict = model(outputs)
        loss = criterion(predict,outputs["label"])
        total_loss += loss.item()
        pred = predict.argmax(dim = 1)
        preds.extend(pred.cpu().numpy())
        labels.extend(outputs["label"].cpu().numpy())
    return {
        "loss":total_loss/len(loader),
        "acc":accuracy_score(labels,preds),
        "precision":precision_score(labels,preds,zero_division = 0),
        "recall":recall_score(labels,preds,zero_division = 0),
        "f1":f1_score(labels,preds,zero_division = 0)}
    
def main():
    seed_everything()
    print("="*60)
    print("Device :", DEVICE)
    print("="*60)
    with open(INDEX_FILE) as f:
        samples = json.load(f)
    labels = [x["label"] for x in samples]
    groups = [x["player"] for x in samples]
    kf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    for fold, (train_idx, val_idx) in enumerate(kf.split(samples, labels, groups),start=1):

        logger.info("\n" + "=" * 80)
        logger.info(f"Fold {fold}/5")
        logger.info("=" * 80)

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_loader, val_loader = build_dataloader(
            train_samples,
            val_samples
        )
        model = build_model()
        freeze_backbone(model)
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
        criterion = nn.CrossEntropyLoss()
        fold_dir = os.path.join(SAVE_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        for epoch in range(EPOCHS):
            print("="*80)
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print("="*80)
            if epoch == 25:
                optimizer = unfreeze_and_update_optimizer(model, optimizer, epoch)
                scheduler = build_scheduler(optimizer, EPOCHS - epoch)
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_metrics = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Train Loss {train_metrics['loss']:.4f} "
                f"Acc {train_metrics['acc']:.4f} "
                f"F1 {train_metrics['f1']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} "
                f"Acc {val_metrics['acc']:.4f} "
                f"F1 {val_metrics['f1']:.4f} | "
                f"LR {current_lr:.6f}"
            )
            print(msg)
            logger.info(msg)
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "f1": val_metrics["f1"],
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            }
            torch.save(checkpoint, os.path.join(fold_dir, f"epoch_{epoch+1:03d}.pt"))
    print()
    print("="*60)
    print("Training Finished")
    print("="*60)

if __name__ == "__main__":
    main()