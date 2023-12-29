from modelling.losses import CLIPLoss, InfoNCE
from logger import log

import torch

def test():
    log("Testing CLIPLoss")
    input1 = torch.rand(16, 128).cuda()
    input2 = torch.rand(16, 128).cuda()
    output = CLIPLoss()(input1, input2)
    log(output)
    log("Testing InfoNCE")
    input1 = torch.rand(16, 128).cuda()
    input2 = torch.rand(16, 128).cuda()
    output = InfoNCE()(input1, input2)
    log(output)





