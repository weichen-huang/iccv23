from dataset import TemporalDataset
from modelling.contrastive import ContrastiveModel
from logger import log
from tqdm import tqdm
import torch
import os


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
        text = f"{self.name}: {self.avg:.4f}"
        return text


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


def main():
    modality_groups = {"dx": ["DX", "ADAS13", "Ventricles", "EXAMDATE"],
                           "cog": ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate"],
                           "vol": ["Hippocampus", "WholeBrain", "Entorhinal", "MidTemp"],
                           "pet": ["FDG", "AV45"],
                           "bio": ["ABETA_UPENNBIOMK9_04_19_17", "TAU_UPENNBIOMK9_04_19_17", "PTAU_UPENNBIOMK9_04_19_17"],
                           "demo": ["APOE4", "AGE"]}
    for modalityb in ["dx", "cog", "vol", "pet", "bio", "demo"]:
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
            "tab_res_disc_shape": (3, ),
            "is_cont": True if modalityb != "dx" else False,

        }

        model = ContrastiveModel(params=params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        step = "epoch"

        train_dataset = TemporalDataset(modality_groups, mode="train", volume=True)
        batch_size = 8
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,

        )

        valid_dataset = TemporalDataset(modality_groups, mode="test", volume=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )

        best_loss = float('inf')
        for epoch in range(64):

            print(f"Epoch: {epoch + 1}")
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, batch_size, params)
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader, batch_size, params)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                model.save("checkpoints/contrast.pt")
                print("Saved Best Model!")

if __name__ == "__main__":
    main()