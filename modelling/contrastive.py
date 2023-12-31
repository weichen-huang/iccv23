import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from modelling import losses
from modelling import utils
from modelling.encoders import ImageEncoder, VolumeEncoder, TabularEncoder, TabTransformerEncoder

from logger import log

class ContrastiveModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        def_params = {
            "img_res": 256,
            "proj_dim": 128,
            "temp": 0.5,
            "modalityb": "dx",
            "modalitytype": "tabular",
            "image_dim": 2,
            "tabmode": "tabtransformer",
            "tab_res_cont": 10,
            "tab_res_disc": 1,
            "tab_res_disc_shape": (3),
            "is_cont": True,

        }

        if params is not None:
            self.params = params
        else:
            self.params = def_params

        if self.params["image_dim"] == 2:
            self.image_encoder = ImageEncoder(input_resolution=self.params["img_res"], output_dim=self.params["proj_dim"]).cuda()
        elif self.params["image_dim"] == 3:
            self.image_encoder = VolumeEncoder(input_resolution=self.params["img_res"], output_dim=self.params["proj_dim"]).cuda()
        else:
            raise ValueError("image_dim must be 2 or 3")

        if self.params["tabmode"] == "tabatt":
            self.tabular_encoder = TabularEncoder(input_resolution=self.params["tab_res_cont"]
                if self.params["is_cont"] else self.params["tab_res_disc"], output_dim=128, training=False).cuda()
        elif self.params["tabmode"] == "tabtransformer":
            self.tabular_encoder = TabTransformerEncoder(
                        categories=self.params["tab_res_disc_shape"] if not self.params["is_cont"] else (),  # tuple containing the number of unique values within each category
                        num_continuous=self.params["tab_res_cont"] if self.params["is_cont"] else 0,  # number of continuous values
                        dim=32,  # dimension, paper set at 32
                        dim_out=self.params["proj_dim"],  # binary prediction, but could be anything
                        depth=6,  # depth, paper recommended 6
                        heads=8,  # heads, paper recommends 8
                        attn_dropout=0.1,  # post-attention dropout
                        ff_dropout=0.1,  # feed forward dropout
                        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
                    ).cuda()
        else:
            raise ValueError("tabmode must be tabatt or tabtransformer")

        self.loss = losses.InfoNCE(temperature=self.params["temp"])

    def forward(self, batch):

        image_embedding = self.image_encoder(batch["image"])
        tabular_embedding = self.forward_tabular(batch[self.params["modalityb"]])

        global_loss = self.loss(image_embedding, tabular_embedding)

        local_loss = 0.0
        identites = list(map(int, batch["ptid"].tolist()))
        image_mapping = {}
        tabular_mapping = {}
        for id in identites:
            if id not in image_mapping:
                image_mapping[id] = []
            if id not in tabular_mapping:
                tabular_mapping[id] = []

        image_embedding = torch.squeeze(image_embedding, dim=1)
        image_embeddings = torch.split(image_embedding, 1, dim=0)
        tabular_embedding = torch.squeeze(tabular_embedding, dim=1)
        tabular_embeddings = torch.split(tabular_embedding, 1, dim=0)

        for i in range(len(identites)):
            image_mapping[identites[i]].append(image_embeddings[i])
            tabular_mapping[identites[i]].append(tabular_embeddings[i])


        for k, v in image_mapping.items():
            if len(v) > 1:
                for i in range(1, len(v)):
                    local_loss += self.loss(v[0], v[i])

        loss = global_loss + local_loss

        return loss, global_loss, local_loss

    def forward_tabular(self, data):
        if self.params["tabmode"] == "tabtransformer":
            if self.params["is_cont"]:
                x_disc = torch.randn(0, 0)
                x_cont = data["tabular_data"]
                return self.tabular_encoder(x_disc, x_cont)
            else:
                x_cont = torch.randn(0, 0)
                x_disc = data["tabular_data"]
                return self.tabular_encoder(x_disc, x_cont)
        else:
            return self.tabular_encoder(data["tabular_data"])
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



