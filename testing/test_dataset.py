from dataset import TemporalDataset
from logger import log

from torch.utils.data import DataLoader

def test():

    # load dataset
    dataset = TemporalDataset()

    # get dataset[0]
    data = dataset[0]

    log(data["volume"].shape)
    log(data["image"].shape)
    log(data["modalityb"])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # get dataloader[0]
    data = next(iter(dataloader))
    log(data)