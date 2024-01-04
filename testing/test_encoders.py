from modelling.encoders import ImageEncoder, VolumeEncoder, TabularEncoder, TabTransformerEncoder, TextEncoder
from logger import log

import torch

def test():

    log("Testing ImageEncoder")
    encoder = ImageEncoder(input_resolution=256, output_dim=128).cuda()
    input = torch.rand(1, 3, 256, 256).cuda()
    output = encoder(input)
    log(output.shape)
    log("Testing VolumeEncoder")
    encoder = VolumeEncoder(input_resolution=256, output_dim=128, n_input_channels=3).cuda()
    input = torch.rand(1, 3, 256, 256, 256).cuda()
    output = encoder(input)
    log(output.shape)
    log("Testing TabularEncoder")
    encoder = TabularEncoder(input_resolution=10, output_dim=128, training=False).cuda()
    input = torch.rand(1, 10).cuda()
    output = encoder(input)
    log(output.shape)
    log("Testing TabTransformerEncoder")

    #categories = (10, 5, 6, 5, 8)
    x_categ = torch.randint(0, 5, (1, 5)).cuda()  # category values, from 0 - max number of categories, in the order as passed into the constructor above
    # print(x_categ)
    x_cont = torch.randn(1, 10).cuda()  # assume continuous values are already normalized individually
    x_categ = torch.randn(0, 0)
    model = TabTransformerEncoder(
        categories=(),  # tuple containing the number of unique values within each category
        num_continuous=10,  # number of continuous values
        dim=32,  # dimension, paper set at 32
        dim_out=128,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
    ).cuda()

    log(x_categ.shape)
    log(x_cont.shape)
    pred = model(x_categ, x_cont).cuda()  # (1, 1)
    log(pred.shape)
    pred = torch.squeeze(pred, dim=1)
    pred = torch.split(pred, 1, dim=0)
    print(pred[0].shape, len(pred))

    log("Testing TextEncoder")
    encoder = TextEncoder(output_dim=128).cuda()
    input = ["I am bert!"]
    output = encoder(input, device=torch.device("cuda"))
    log(output.shape)

if __name__ == "__main__":
    test()

