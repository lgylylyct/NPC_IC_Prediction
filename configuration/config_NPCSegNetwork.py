class Configuration:
    def __init__(self) -> None:
        pass

    def to_dict(self):
        return self.__dict__


cfg_seg = Configuration()

# GTV Segmentation
cfg_seg.MRI_sequences = ["T1C", "GTV_mask"]
cfg_seg.seg_classes = ["BG", "GTV"]

# MLN
# cfg_seg.MRI_sequences = ["T2", "MLN_mask"]
# cfg_seg.seg_classes = ["BG", "MLN"]

cfg_seg.layer_channels = [32, 32, 64, 64, 128, 128]
cfg_seg.channel_rate = 1
