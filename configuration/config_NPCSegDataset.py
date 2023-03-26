class Configuration:
    def __init__(self) -> None:
        pass

    def to_dict(self):
        return self.__dict__


cfg_seg_data = Configuration()

cfg_seg_data.crop_shape = [32, 192, 192]
cfg_seg_data.ds_rate = (8, 32, 32)  # downsample rate


# cfg_seg_data.datalist_path = "DataList/Datalist_NPC_Seg_GTV.json"
# cfg_seg_data.datalist_split_path = "DataList/Datasplit_NPC_Seg_GTV.json"
# cfg_seg_data.MRI_sequences = ["T1C","GTV_mask"]
# cfg_seg_data.seg_classes = ["BG", "GTV"]


cfg_seg_data.datalist_path = "DataList/Datalist_NPC_Seg_MLN.json"
cfg_seg_data.datalist_split_path = "DataList/Datasplit_NPC_Seg_MLN.json"
cfg_seg_data.MRI_sequences = ["T2", "MLN_mask"]
cfg_seg_data.seg_classes = ["BG", "MLN"]

# path
cfg_seg_data.virtual_path = "DataSet"
cfg_seg_data.practical_path = "DataSet"





cfg_seg_data.exclude_patients_id = []




# Data Augmentation
cfg_seg_data.aug_translation = 1
cfg_seg_data.aug_rotate = 1

cfg_seg_data.aug_intensity_gamma = 1
cfg_seg_data.aug_intensity_shift = 1
cfg_seg_data.aug_intensity_scale = 1

cfg_seg_data.aug_gaussian_noise = 1
cfg_seg_data.aug_gaussian_smooth = 1


# iteration configuration
cfg_seg_data.current_KF = 1
cfg_seg_data.end_KF = 5
cfg_seg_data.num_KF = 5
cfg_seg_data.current_trial = 1
cfg_seg_data.num_trial = 1


# other
cfg_seg_data.trainset = "train"
cfg_seg_data.valset = "val"
cfg_seg_data.testset = "val"

cfg_seg_data.buffer = False
