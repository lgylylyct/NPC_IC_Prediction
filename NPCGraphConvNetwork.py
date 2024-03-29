import torch
from gcn_lib import (
    BasicConv,
    GraphConv2d,
    PlainDynBlock2d,
    ResDynBlock2d,
    DenseDynBlock2d,
    DenseDilatedKnnGraph,
)
from torch.nn import Sequential as Seq
import torch.nn as nn
from math import cos, pi


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, cfg, in_channels, logger=None):
        super(DenseDeepGCN, self).__init__()
        self.logger = logger
        channels = cfg.n_filters
        k = cfg.k
        act = cfg.act
        norm = cfg.norm
        bias = cfg.bias
        epsilon = cfg.epsilon
        stochastic = cfg.stochastic
        conv = cfg.conv
        c_growth = channels
        self.n_blocks = cfg.n_blocks
        grow_num = cfg.grow_num
        if grow_num == 0:
            stochastic = False

        self.in_channels = in_channels
        self.schedule_cat_clinic = cfg.schedule_cat_clinic
        self.clinic_channels = cfg.clinic_channels
        if self.schedule_cat_clinic in ["tail", "knn"]:
            self.in_channels = self.in_channels - self.clinic_channels

        self.embedding = torch.nn.Linear(self.in_channels, channels)
        self.relu = torch.nn.ReLU()

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(channels, channels, conv, act, norm, bias)

        dilate = 1
        seq = []
        for i in range(self.n_blocks - 1):
            if cfg.block.lower() == "res":
                seq.append(
                    ResDynBlock2d(
                        channels, k, int(dilate), conv, act, norm, bias, stochastic, epsilon,
                    )
                )
            elif cfg.block.lower() == "dense":
                seq.append(
                    DenseDynBlock2d(
                        channels + c_growth * i,
                        c_growth,
                        k,
                        int(dilate),
                        conv,
                        act,
                        norm,
                        bias,
                        stochastic,
                        epsilon,
                    )
                )
            else:
                seq.append(
                    PlainDynBlock2d(
                        channels, k, int(dilate), conv, act, norm, bias, stochastic, epsilon,
                    )
                )
            dilate += grow_num
        self.backbone = Seq(*seq)

        if cfg.block.lower() == "res":
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif cfg.block.lower() == "dense":
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2
            )
        else:
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_dims = fusion_dims

        self._init_network(cfg.init_strategy)

    def forward(self, inputs):
        if self.schedule_cat_clinic in ["tail", "knn"]:
            feas_other, feas_clinic = (
                inputs[:, self.clinic_channels :],
                inputs[:, : self.clinic_channels],
            )
            feas_clinic = feas_clinic[None].transpose(1, 2).unsqueeze(-1)
        else:
            feas_other = inputs

        embedding = self.embedding(feas_other)  # (N,C)
        embedding = self.relu(embedding)
        embedding = embedding[None].transpose(1, 2).unsqueeze(-1)  # (B,C,N,1)

        if self.schedule_cat_clinic == "knn":
            feas = [self.head(embedding, self.knn(feas_clinic))]
        else:
            feas = [self.head(embedding, self.knn(embedding))]

        for i in range(self.n_blocks - 1):
            feas.append(self.backbone[i](feas[-1]))
        feas = torch.cat(feas, dim=1)
        feas = feas.squeeze(-1).transpose(1, 2)[0]

        return feas

    def _init_network(self, init_strategy):
        if init_strategy is None:
            if self.logger is not None:
                self.logger.warning("No initialization")
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        elif init_strategy == "trunc_normal":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if self.logger is not None:
                        self.logger.warning("trunc_normal")
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        elif "kaiming_normal" in init_strategy:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if init_strategy == "kaiming_normal2":
                        if self.logger is not None:
                            self.logger.warning("kaiming_normal2")
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    else:
                        if self.logger is not None:
                            self.logger.warning("kaiming_normal")
                        nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            assert False, "init_strategy is {}, must be ['trunc_normal','kaiming_normal']"


class Projection_MLP(torch.nn.Module):
    def __init__(
        self, in_dims, hidden_dims, out_dims, num_layers, norm, act, last_norm, dropout=0.0,
    ):
        super(Projection_MLP, self).__init__()

        self.in_dims = in_dims
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * (num_layers - 1)
        else:
            self.hidden_dims = hidden_dims

        self.out_dims = out_dims
        self.num_layers = num_layers

        if norm == "bn":
            cls_norm = nn.BatchNorm1d
            self.bias = False
        else:
            cls_norm = nn.Identity
            self.bias = True

        if act == "relu":
            cls_act = nn.ReLU
        else:
            cls_act = nn.Identity

        if last_norm == "bn":
            cls_last_norm = nn.BatchNorm1d
            self.last_bias = False
        else:
            cls_last_norm = nn.Identity
            self.last_bias = True

        self.layer1 = nn.Sequential(
            nn.Linear(self.in_dims, self.hidden_dims[0], bias=self.bias),
            cls_norm(self.hidden_dims[0]),
            cls_act(),
            nn.Dropout(dropout),
        )
        for i in range(1, self.num_layers - 1):
            setattr(
                self,
                "layer{}".format(i + 1),
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i], bias=self.bias),
                    cls_norm(self.hidden_dims[i]),
                    cls_act(),
                    nn.Dropout(dropout),
                ),
            )
        self.lase_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.out_dims, bias=self.last_bias),
            cls_last_norm(self.out_dims),
        )

    def forward(self, x):
        x = self.layer1(x)
        for i in range(1, self.num_layers - 1):
            x = getattr(self, "layer{}".format(i + 1))(x)
        x = self.lase_layer(x)

        return x


class GCN_NPCIC(torch.nn.Module):
    def __init__(self, cfg, in_channels, logger=None):
        super(GCN_NPCIC, self).__init__()
        self.logger = logger
        self.schedule_cat_clinic = cfg.schedule_cat_clinic

        self.encoder = DenseDeepGCN(cfg, in_channels, logger)

        self.dropout = torch.nn.Dropout(p=cfg.dropout)

        if self.schedule_cat_clinic in ["tail"]:
            self.fusion_dims = self.encoder.fusion_dims + self.clinic_channels

        else:
            self.fusion_dims = self.encoder.fusion_dims

        if cfg.add_pdt:
            self.tail = Projection_MLP(
                self.fusion_dims,
                cfg.pdc_hidden_dims,
                cfg.n_classes,
                cfg.pdc_num_layers,
                cfg.pdc_norm,
                cfg.pdc_atc,
                cfg.pdc_last_norm,
                cfg.dropout,
            )
        else:
            self.tail = nn.Linear(self.fusion_dims, cfg.n_classes, bias=False)

    def forward(self, inputs):
        feas = self.encoder(inputs)
        feas = self.dropout(feas)

        if self.schedule_cat_clinic == "tail":
            feas_clinic = inputs[:, : self.clinic_channels]
            feas = torch.cat([feas, feas_clinic], dim=1)

        scores = self.tail(feas)

        return scores


from configuration.config_NPCGraphConvNetwork import cfg_gcn

if __name__ == "__main__":
    num_patients, dim_features = (100, 23)
    inputs = torch.randn(size=(num_patients, dim_features))
    gcn_model = GCN_NPCIC(cfg=cfg_gcn, in_channels=dim_features).eval()
    with torch.no_grad():
        scores = gcn_model(inputs)
    print(scores.shape)
