import os
import platform
import random, json
import numpy as np
import torch
import torch.utils.data as data
from scipy import ndimage


def readJson(path):
    with open(path, "r") as f:
        data_dict = dict(json.load(f))
    return data_dict


def toTensor(ndarray, dtype=torch.float32, add_channel=0):
    if not isinstance(ndarray, np.ndarray):
        return ndarray

    for i in range(add_channel):
        ndarray = ndarray[np.newaxis, ...]
    tensor = torch.as_tensor(ndarray.copy(), dtype=dtype)
    return tensor


class ObjectDataset(data.Dataset):
    def __init__(self, train_val_test, cfg):
        self.train_val_test = train_val_test
        self.cfg = cfg
        self.buffer = {}

    def show_buffer_info(self):
        if len(self.buffer) % (len(self) // 4) == 0:
            print("buffer loading: {}/{} ......".format(len(self.buffer), len(self)))
        elif len(self.buffer) == len(self):
            print("buffer loading: {}/{} ......".format(len(self.buffer), len(self)))

    def padding_in_test(self, img):
        C, D, H, W = img.shape
        padding_d = np.ceil(D / self.cfg.ds_rate[0]) * self.cfg.ds_rate[0]
        padding_h = np.ceil(H / self.cfg.ds_rate[1]) * self.cfg.ds_rate[1]
        padding_w = np.ceil(W / self.cfg.ds_rate[2]) * self.cfg.ds_rate[2]

        img = np.pad(
            img,
            (
                (0, 0),
                (0, int(padding_d) - D),
                (0, int(padding_h) - H),
                (0, int(padding_w) - W),
            ),
            mode="minimum",
        )
        origin_DHW = (D, H, W)

        return img, origin_DHW


class NPCSegDataset(ObjectDataset):
    def __len__(self):
        return len(self.patients_id)

    def __init__(self, train_val_test, cfg):
        super().__init__(train_val_test, cfg)
        self.seg_classes = self.cfg.seg_classes

        path_datalist = self.cfg.datalist_path
        assert os.path.isfile(path_datalist), path_datalist + " is not exist"
        self.dict_datalist = readJson(path_datalist)

        path_datasplit = self.cfg.datalist_split_path
        assert os.path.isfile(path_datalist), path_datalist + " is not exist"
        self.dict_datasplit = readJson(path_datasplit)

        suffix = str(self.cfg.current_KF) if self.cfg.num_KF != 1 else ""
        dataset_name = getattr(self.cfg, train_val_test + "set") + suffix

        patients_id = self.dict_datasplit[dataset_name]

        ########## exclude patinet ##########
        for ex in self.cfg.exclude_patients_id:
            if ex in patients_id:
                index = patients_id.index(ex)
                patients_id.pop(index)

        self.patients_id = []
        ########## load path ##########
        for patient_id in patients_id:
            self.patients_id.append(patient_id)

        if self.cfg.buffer:
            for i in range(len(self)):
                self.load_img_label(i)
                self.show_buffer_info()

        print()

    def load_img_label(self, index):

        patient_id = self.patients_id[index]
        key = "{}".format(patient_id)

        # load from buffer
        if self.cfg.buffer:
            if key in self.buffer:
                img, label, patient_id = self.buffer[key]
                return img, label, patient_id

        # load img
        img = []
        for s in self.cfg.MRI_sequences:
            if "mask" in s:
                continue
            path_img = self.dict_datalist[patient_id][s]
            img.append(np.load(path_img))
        img = np.stack(img)

        # load mask
        mask = []
        for s in self.cfg.MRI_sequences:
            if "mask" not in s:
                continue
            path_mask = self.dict_datalist[patient_id][s]
            mask.append(np.load(path_mask))

        label = np.zeros_like(img[0])
        for i in range(len(mask)):
            label[mask[i] != 0] = i + 1

        # load into buffer
        if self.cfg.buffer:
            self.buffer[key] = [img, label, patient_id]

        return img, label, patient_id

    def cropByBox(self, patient_id, img, label):
        d, h, w = self.cfg.crop_shape
        C, D, H, W = img.shape
        padding_d = d - D if D < d else 0
        padding_h = h - H if H < h else 0
        padding_w = w - W if W < w else 0

        img = np.pad(
            img, ((0, 0), (0, padding_d), (0, padding_h), (0, padding_w)), mode="minimum"
        )
        label = np.pad(
            label, ((0, padding_d), (0, padding_h), (0, padding_w)), mode="constant"
        )
        C, new_D, new_H, new_W = img.shape

        if "T1" in self.cfg.MRI_sequences and "T2" in self.cfg.MRI_sequences:
            assert False
        elif "T1C" in self.cfg.MRI_sequences:
            bbox = self.dict_datalist[patient_id]["GTV_mask_bbox"]
        elif "T2" in self.cfg.MRI_sequences:
            bbox = self.dict_datalist[patient_id]["MLN_mask_bbox"]
        d1, h1, w1, d2, h2, w2 = bbox

        crop_d1, crop_h1, crop_w1 = 0, 0, 0

        # translation augmentation
        if random.random() < self.cfg.aug_translation:
            if self.train_val_test == "train":
                offset_d = d - (d2 - d1)
                offset_h = h - (h2 - h1)
                offset_w = w - (w2 - w1)
                offset_d = random.randint(0, offset_d)
                offset_h = random.randint(0, offset_h)
                offset_w = random.randint(0, offset_w)
                crop_d1, crop_h1, crop_w1 = d1 - offset_d, h1 - offset_h, w1 - offset_w

        if crop_d1 < 0:
            crop_d1 = 0
        if crop_h1 < 0:
            crop_h1 = 0
        if crop_w1 < 0:
            crop_w1 = 0

        crop_d2 = crop_d1 + d
        crop_h2 = crop_h1 + h
        crop_w2 = crop_w1 + w

        if crop_d2 > new_D:
            crop_d2 = new_D
            crop_d1 = crop_d2 - d

        if crop_h2 > new_H:
            crop_h2 = new_H
            crop_h1 = crop_h2 - h

        if crop_w2 > new_W:
            crop_w2 = new_W
            crop_w1 = crop_w2 - w

        # rotation augmentation
        if random.random() < self.cfg.aug_rotate:
            if self.train_val_test == "train":
                angle = random.randint(-10, 10)
                img = ndimage.rotate(
                    img, angle=angle, axes=(2, 3), reshape=False, order=2, mode="nearest",
                )

                label = ndimage.rotate(
                    label, angle=angle, axes=(1, 2), reshape=False, order=0, mode="nearest",
                )

        img = img[:, crop_d1:crop_d2, crop_h1:crop_h2, crop_w1:crop_w2]
        label = label[crop_d1:crop_d2, crop_h1:crop_h2, crop_w1:crop_w2]

        if img.shape[1:] != (d, h, w):
            print(img.shape)
            print(patient_id)
            print(crop_d1, crop_d2, crop_h1, crop_h2, crop_w1, crop_w2)

        return img, label

    def run_augmentation(self, img, label):
        if self.train_val_test != "train":
            return img, label

        aug_img = img.copy()
        aug_label = label.copy()
        C, D, H, W = aug_img.shape

        # intensity augmentation
        if random.random() < self.cfg.aug_intensity_gamma:
            gamma = random.uniform(0.7, 1.3)
            for c in range(C):
                temp = aug_img[c]
                temp_min = temp.min()
                temp_max = temp.max()
                temp = (temp - temp_min) / (temp_max - temp_min)
                temp = temp ** gamma
                temp = temp * (temp_max - temp_min) + temp_min
                aug_img[c] = temp

        if random.random() < self.cfg.aug_intensity_shift:
            offset = random.uniform(-0.5, 0.5)
            aug_img = aug_img - offset

        if random.random() < self.cfg.aug_intensity_scale:
            scale = random.uniform(0.8, 1.2)
            aug_img = aug_img * scale

        # perturbation augmentation
        if random.random() < 0.5:
            if random.random() < self.cfg.aug_gaussian_noise:
                mean = 0
                sigma = random.uniform(0, 0.15)
                gauss = np.random.normal(mean, sigma, aug_img.shape)
                aug_img += gauss
        else:
            if random.random() < self.cfg.aug_gaussian_smooth:
                sigma = random.uniform(0, 0.5)
                for c in range(C):
                    aug_img[c] = ndimage.gaussian_filter(aug_img[c], (0, sigma, sigma))

        return aug_img, aug_label

    def __getitem__(self, index):
        img, label, patient_id = self.load_img_label(index)

        if self.train_val_test != "test":
            aug_img, aug_label = self.cropByBox(patient_id, img, label)
            aug_img, aug_label = self.run_augmentation(aug_img, aug_label)

            inputs = toTensor(aug_img)
            targets = toTensor(aug_label, torch.long)
            return inputs, targets, patient_id
        else:
            padding_img, origin_DHW = self.padding_in_test(img)
            inputs = toTensor(padding_img)
            targets = toTensor(label, torch.long)
            return inputs, targets, origin_DHW, patient_id


from configuration.config_NPCSegDataset import cfg_seg_data

if __name__ == "__main__":
    train_dataset = NPCSegDataset(train_val_test="train", cfg=cfg_seg_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    for inputs, targets, patient_id in train_dataloader:
        print(inputs.shape, targets.shape, patient_id)

    test_dataset = NPCSegDataset(train_val_test="test", cfg=cfg_seg_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    for inputs, targets, origin_DHW, patient_id in test_dataloader:
        print(inputs.shape, targets.shape, origin_DHW, patient_id)

