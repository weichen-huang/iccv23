import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import TemporalDataset
from modelling.contrastive import ContrastiveModel
from scripts.validation import CrossValidator, EarlyStopping, MetricsTracker
from logger import log
from tqdm import tqdm
import torch


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, batch_size, params):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        batch["image"] = batch["image"].cuda()
        if params["modalityb"] == "dx":
            batch[params["modalityb"]] = batch[params["modalityb"]][0].cuda().long().unsqueeze(-1)
        batch[params["modalityb"]] = batch[params["modalityb"]].cuda()
        loss, _, _ = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        loss_meter.update(loss.item(), batch_size)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader, batch_size, params):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    for batch in tqdm_object:
        batch["image"] = batch["image"].cuda()
        if params["modalityb"] == "dx":
            batch[params["modalityb"]] = batch[params["modalityb"]][0].cuda().long().unsqueeze(-1)
        batch[params["modalityb"]] = batch[params["modalityb"]].cuda()
        loss, _, _ = model(batch)

        loss_meter.update(loss.item(), batch_size)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter


def get_modality_groups():
    return {
        "dx": ["DX", "ADAS13", "Ventricles", "EXAMDATE"],
        "cog": ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate"],
        "vol": ["Hippocampus", "WholeBrain", "Entorhinal", "MidTemp"],
        "pet": ["FDG", "AV45"],
        "bio": ["ABETA_UPENNBIOMK9_04_19_17", "TAU_UPENNBIOMK9_04_19_17", "PTAU_UPENNBIOMK9_04_19_17"],
        "demo": ["APOE4", "AGE"]
    }


def train_single_modality(modalityb, args, modality_groups):
    log(f"Modality: {modalityb}")

    params = {
        "img_res": 196,
        "proj_dim": 128,
        "temp": 0.5,
        "modalityb": modalityb,
        "modalitytype": "tabular",
        "image_dim": 3,
        "tabmode": "tabtransformer",
        "tab_res_cont": len(modality_groups[modalityb]),
        "tab_res_disc": 1,
        "tab_res_disc_shape": (3,),
        "is_cont": True if modalityb != "dx" else False,
    }

    batch_size = args.batch_size

    if args.cross_validation:
        log(f"Running {args.n_folds}-fold cross-validation")

        full_dataset = TemporalDataset(modality_groups, mode="train", volume=True)
        full_data = full_dataset.data

        cv = CrossValidator(
            n_folds=args.n_folds,
            early_stopping_patience=args.early_stopping_patience,
            output_dir=os.path.join(args.output_dir, modalityb),
            stratified=True
        )

        def model_factory():
            return ContrastiveModel(params=params)

        def train_fn(model, train_data, epoch):
            train_dataset = TemporalDataset(
                modality_groups,
                modalityb=modalityb,
                volume=True,
                fold_indices=list(range(len(train_data)))
            )
            train_dataset.data = train_data

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            model.train()
            loss_meter = train_epoch(model, train_loader, optimizer, None, "epoch", batch_size, params)
            return loss_meter.avg

        def val_fn(model, val_data):
            val_dataset = TemporalDataset(
                modality_groups,
                modalityb=modalityb,
                volume=True,
                fold_indices=list(range(len(val_data)))
            )
            val_dataset.data = val_data

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )

            model.eval()
            with torch.no_grad():
                loss_meter = valid_epoch(model, val_loader, batch_size, params)
            return loss_meter.avg

        results = cv.run(
            data=full_data,
            model_factory=model_factory,
            train_fn=train_fn,
            val_fn=val_fn,
            n_epochs=args.epochs,
            verbose=True
        )

        return results

    else:
        model = ContrastiveModel(params=params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        step = "epoch"

        train_dataset = TemporalDataset(modality_groups, mode="train", volume=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        valid_dataset = TemporalDataset(modality_groups, mode="test", volume=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        early_stopping = EarlyStopping(patience=args.early_stopping_patience)
        metrics = MetricsTracker(output_dir=args.output_dir)
        fold_metrics = metrics.add_fold(0)

        best_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch + 1}")
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, batch_size, params)
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader, batch_size, params)

            fold_metrics.update(train_loss.avg, valid_loss.avg, epoch)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                os.makedirs(args.output_dir, exist_ok=True)
                model.save(os.path.join(args.output_dir, f"{modalityb}_best.pt"))
                print("Saved Best Model!")

            if early_stopping(valid_loss.avg):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        metrics.save(f"{modalityb}_metrics.json")
        return metrics.get_summary()


def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Contrastive Learning Training')
    parser.add_argument('--epochs', type=int, default=64, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--cross_validation', action='store_true', help='Enable K-fold cross-validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--modality', type=str, default='all',
                        choices=['all', 'dx', 'cog', 'vol', 'pet', 'bio', 'demo'],
                        help='Modality to train (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()
    modality_groups = get_modality_groups()

    if args.modality == 'all':
        modalities = ["dx", "cog", "vol", "pet", "bio", "demo"]
    else:
        modalities = [args.modality]

    all_results = {}
    for modalityb in modalities:
        results = train_single_modality(modalityb, args, modality_groups)
        all_results[modalityb] = results

    log("Training complete!")
    return all_results


if __name__ == "__main__":
    main()