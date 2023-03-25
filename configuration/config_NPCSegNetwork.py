class Configuration:
    def __init__(self) -> None:
        pass

    def to_dict(self):
        return self.__dict__


cfg_seg = Configuration()

# GTV Segmentation
cfg_seg.MRI_sequences = ["T1C", "T1C_mask"]
cfg_seg.seg_classes = ["BG", "Tumor"]

# MLN
# cfg_seg.MRI_sequences = ["T2", "T2_mask"]
# cfg_seg.seg_classes = ["BG", "Lymph"]

cfg_seg.layer_channels = [32, 32, 64, 64, 128, 128]
cfg_seg.channel_rate = 1
