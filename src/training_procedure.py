import os

import torch
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from .loss_utils import get_loss, SupConLoss
from .stat_keeper import StatKeeper


def stat_step(preds, gts, stat_keeper):
    """Accumulate confusion-matrix-based statistics for a single step."""
    binary_preds = preds.detach().cpu().numpy()
    binary_gts = gts.detach().cpu().numpy()
    stat_keeper.step(binary_preds, binary_gts)


def apply_cutmix(images, label1, label2=None, alpha=1.0):
    """Perform CutMix augmentation (optionally on a 2-headed setup)."""
    batch_size, C, H, W = images.size()
    indices = torch.randperm(batch_size).to(images.device)
    shuffled_images = images[indices]
    shuffled_label1 = label1[indices]
    if label2 is not None:
        shuffled_label2 = label2[indices]

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    new_label1 = lam * label1 + (1 - lam) * shuffled_label1
    if label2 is not None:
        new_label2 = lam * label2 + (1 - lam) * shuffled_label2
        return images, new_label1, new_label2
    return images, new_label1


class Trainer:
    """Generic trainer that handles CutMix, SupCon, W&B logging and model export."""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        schedulers,
        device,
        epochs,
        weights_save_folder,
        supcon_weight: float = 0.1,
        use_multihead: bool = False,
        output_name: str = None,
        config=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.weights_save_folder = weights_save_folder
        self.use_multihead = use_multihead
        self.output_name = output_name
        self.config = config

        self.pt_path = os.path.join(weights_save_folder, "pt")
        self.onnx_path = os.path.join(weights_save_folder, "onnx")
        self.openvino_path = os.path.join(weights_save_folder, "openvino")

        os.makedirs(weights_save_folder, exist_ok=True)
        os.makedirs(self.pt_path, exist_ok=True)
        os.makedirs(self.onnx_path, exist_ok=True)
        os.makedirs(self.openvino_path, exist_ok=True)

        self.supcon = (
            SupConLoss(temperature=config.loss.get("supcon_temperature", 0.07))
            if supcon_weight
            else None
        )
        self.sup_w = supcon_weight

    def get_lr(self):
        """Return the LR of the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, dataloader):
        """Run one training epoch."""
        self.model.train()
        accum_loss = 0.0
        stat_keeper = StatKeeper(self.config)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
            images = batch[0].to(self.device).float()
            labels_true = batch[1].to(self.device).float()
            self.optimizer.zero_grad()

            # CutMix
            if np.random.rand() < getattr(self.config, "cutmix_prob", 0.5):
                if self.use_multihead:
                    aux_labels = batch[2].to(self.device).float()
                    images, labels_true, aux_labels = apply_cutmix(
                        images,
                        labels_true,
                        aux_labels,
                        alpha=getattr(self.config, "cutmix_alpha", 1.0),
                    )
                else:
                    images, labels_true = apply_cutmix(
                        images,
                        labels_true,
                        alpha=getattr(self.config, "cutmix_alpha", 1.0),
                    )

            # Forward
            if self.supcon:
                labels_pred, feats = self.model(images, return_feats=True)
            else:
                labels_pred = self.model(images)

            loss_dict = get_loss(
                labels_pred=labels_pred[:, 0],
                labels_true=labels_true,
                criterion=self.criterion,
            )

            if self.supcon:
                con_loss = self.supcon(feats, labels_true.long())
                loss_dict["supcon"] = self.sup_w * con_loss
            loss = sum(loss_dict.values())

            # Statistics
            if batch_idx % 10 == 0:
                stat_step(labels_pred.clone(), labels_true.clone(), stat_keeper)

            loss.backward()
            self.optimizer.step()
            for s in self.schedulers:
                s.step()
            accum_loss += loss.item()

        stats = stat_keeper.get_stat()
        stats["loss"] = accum_loss / len(dataloader)
        return stats

    def val_model(self, dataloader, split_name: str = "val", thr_for_acer: float = None):
        """Run validation."""
        self.model.eval()
        stat_keeper = StatKeeper(self.config)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval-{split_name}", leave=False):
                images = batch[0].to(self.device).float()
                labels_true = batch[1].to(self.device).float()
                preds = (
                    self.model(images, training_mode=False)[0]
                    if self.use_multihead
                    else self.model(images)
                )
                stat_step(preds, labels_true, stat_keeper)

        metrics = stat_keeper.get_stat()
        if thr_for_acer is not None:
            apcer, bpcer, acer = stat_keeper.rates_at_thr(thr_for_acer)
            metrics.update({
                "apcer_val_eer_thr": apcer,
                "bpcer_val_eer_thr": bpcer,
                "acer_val_eer_thr": acer,
            })
        return metrics

    def run(self, train_loader, val_loader):
        """Main training loop."""
        best_eer = [100.0]

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.val_model(val_loader, split_name="val")

            self.log_and_display_metrics(train_metrics, val_metrics, epoch)
            self.save_model(val_metrics, best_eer)

    def log_and_display_metrics(self, train_metrics, val_metrics, epoch):
        """Log metrics to W&B and display in console."""
        metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        metrics["lr"] = self.get_lr()
        metrics["epoch"] = epoch

        if self.config.wandb.get("use", False):
            import wandb
            wandb.log(metrics)

        headers = ["split", *list(val_metrics.keys()), "train_loss"]
        rows = [["val", *list(val_metrics.values()), train_metrics["loss"]]]
        print(tabulate([headers] + rows))

    def save_model(self, val_metrics: dict, best_eer: list):
        """Save model checkpoint if EER improved."""
        val_eer = val_metrics.get("eer", 0.0)

        if val_eer <= best_eer[0]:
            best_eer[0] = val_eer
            filename = f"{self.output_name or 'best'}_val{val_eer:.4f}.pth"
            torch.save(self.model.state_dict(), os.path.join(self.pt_path, filename))
            print(f"Saved new best model → {filename} (EER {val_eer:.4f})")
