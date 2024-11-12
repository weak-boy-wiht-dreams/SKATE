import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):
        
        self.params = {
            'debug': debug,
            'data_path': data_path,
            'label_path': label_path,
            'split': split,
            'data_type': data_type,
            'aug_method': aug_method,
            'intra_p': intra_p,
            'inter_p': inter_p,
            'window_size': window_size,
            'p_interval': p_interval,
            'thres': thres,
            'uniform': uniform,
            'partition': partition
        }
        for key, value in self.params.items():
            setattr(self, key, value)
        
        self.load_data()
        self.init_partition() if partition else None

    def load_data(self):
        npz_data = np.load(self.data_path)
        data_key = 'x_train' if self.split == 'train' else 'x_test'
        label_key = 'y_train' if self.split == 'train' else 'y_test'
        self.data = npz_data[data_key]
        self.label = np.where(npz_data[label_key] > 0)[1]
        self.sample_name = [f"{self.split}_{i}" for i in range(len(self.data))]
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def init_partition(self):
        # Define body part indices for partition
        self.right_arm = np.array([7, 8, 22, 23]) - 1
        self.left_arm = np.array([11, 12, 24, 25]) - 1
        self.right_leg = np.array([13, 14, 15, 16]) - 1
        self.left_leg = np.array([17, 18, 19, 20]) - 1
        self.h_torso = np.array([5, 9, 6, 10]) - 1
        self.w_torso = np.array([2, 3, 1, 4]) - 1
        self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy, label = self.data[index].copy(), self.label[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Apply cropping based on uniformity
        crop_func = tools.valid_crop_uniform if self.uniform else tools.valid_crop_resize
        data_numpy, index_t = crop_func(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)
        
        # Apply augmentations if training
        if self.split == 'train':
            data_numpy = self.apply_augmentations(data_numpy, label, index_t)
        
        # Convert to different data types
        data_numpy = self.apply_data_type(data_numpy)
        
        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        return data_numpy, index_t, label, index

    def apply_augmentations(self, data_numpy, label, index_t):
        p = np.random.rand(1)
        if p < self.intra_p:
            data_numpy = self.intra_instance_augment(data_numpy)
        elif p < (self.intra_p + self.inter_p):
            data_numpy = self.inter_instance_augment(data_numpy, label, index_t)
        return data_numpy

    def intra_instance_augment(self, data_numpy):
        for method, func in [
            ('a', lambda x: x[:, :, :, [1, 0]] if np.random.rand() < 0.5 else x),
            ('b', lambda x: tools.nullify_random_person(x) if np.random.rand() < 0.5 else x),
            ('1', tools.shear),
            ('2', tools.rotate),
            ('3', tools.scale),
            ('4', tools.spatial_flip),
            ('5', tools.temporal_flip),
            ('6', tools.gaussian_noise),
            ('7', tools.gaussian_filter),
            ('8', tools.drop_axis),
            ('9', tools.drop_joint)
        ]:
            if method in self.aug_method:
                data_numpy = func(data_numpy, p=0.5)
        return data_numpy

    def inter_instance_augment(self, data_numpy, label, index_t):
        adain_idx = random.choice(np.where(self.label == label)[0])
        data_adain = self.data[adain_idx].copy()
        f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
        t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int)
        data_adain = data_adain[:, t_idx]
        return tools.skeleton_adain_bone_length(data_numpy, data_adain)

    def apply_data_type(self, data_numpy):
        if self.data_type == 'b':
            data_numpy = tools.joint2bone()(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools.to_motion(data_numpy)
        elif self.data_type == 'bm':
            data_numpy = tools.to_motion(tools.joint2bone()(data_numpy))
        return data_numpy

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

