import torch

from configuration.config_NPCSegNetwork import cfg_seg
from NPCSegNetwork import NNUnetNPC

from configuration.config_NPCGraphConvNetwork import cfg_gcn
from NPCGraphConvNetwork import GCN_NPCIC

# segmentation model for NPC GTV and MLN
D, W, H = (32, 192, 192)
inputs = torch.randn(size=(1, 1, D, W, H))
seg_model = NNUnetNPC(cfg=cfg_seg).eval()
with torch.no_grad():
    segmentation = seg_model(inputs).squeeze().argmax(dim=0)
print(segmentation.shape)

# dynamic GCN for NPC IC prediction
num_patients, dim_features = (100, 23)
inputs = torch.randn(size=(num_patients, dim_features))
gcn_model = GCN_NPCIC(cfg=cfg_gcn, in_channels=dim_features).eval()
with torch.no_grad():
    scores = gcn_model(inputs)
print(scores.shape)
