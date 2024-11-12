import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):
        self.config = {
            'data_path': data_path, 'label_path': label_path, 'split': split,
            'data_type': data_type, 'aug_method': aug_method, 'intra_p': intra_p,
            'inter_p': inter_p, 'window_size': window_size, 'p_interval': p_interval,
            'debug': debug, 'thres': thres, 'uniform': uniform, 'partition': partition
        }
        for key, value in self.config.items():
            setattr(self, key, value)
        
        self.load_data()
    
    def load_data(self):
        """Loads data and labels based on the split type."""
        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)
        prefix = 'train' if self.split == 'train' else 'test'
        self.sample_name = [f'{prefix}_{i}' for i in range(len(self.data))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy, label = np.array(self.data[index]), self.label[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy, index_t = self.apply_cropping(data_numpy, valid_frame_num)
        
        # Apply augmentations if training
        if self.split == 'train':
            data_numpy = self.apply_augmentations(data_numpy, label, index_t)
        
        # Apply data type transformations
        data_numpy = self.apply_data_type(data_numpy)
        
        return data_numpy, index_t, label, index

    def apply_cropping(self, data_numpy, valid_frame_num):
        """Applies cropping or resizing to the data."""
        crop_func = tools.valid_crop_uniform if self.uniform else tools.valid_crop_resize
        return crop_func(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)

    def apply_augmentations(self, data_numpy, label, index_t):
        """Applies intra-instance and inter-instance augmentations."""
        p = np.random.rand()
        if p < self.intra_p:
            return self.intra_instance_augment(data_numpy)
        elif p < self.intra_p + self.inter_p:
            return self.inter_instance_augment(data_numpy, label, index_t)
        return data_numpy

    def intra_instance_augment(self, data_numpy):
        """Applies intra-instance augmentations based on the selected methods."""
        augmentations = {
            'a': lambda x: x[:, :, :, [1, 0]] if np.random.rand() < 0.5 else x,
            'b': self.random_person_zero,
            '1': tools.shear, '2': tools.rotate, '3': tools.scale,
            '4': tools.spatial_flip, '5': tools.temporal_flip,
            '6': tools.gaussian_noise, '7': tools.gaussian_filter,
            '8': tools.drop_axis, '9': tools.drop_joint
        }
        for method, func in augmentations.items():
            if method in self.aug_method:
                data_numpy = func(data_numpy, p=0.5)
        return data_numpy

    def random_person_zero(self, data_numpy):
        """Randomly zeroes out data for a person."""
        if np.random.rand() < 0.5:
            C, T, V, M = data_numpy.shape
            axis_next = np.random.randint(0, 2)
            data_numpy[:, :, :, axis_next] = np.zeros((C, T, V))
        return data_numpy

    def inter_instance_augment(self, data_numpy, label, index_t):
        """Applies inter-instance augmentation."""
        adain_idx = random.choice(np.where(self.label == label)[0])
        data_adain = np.array(self.data[adain_idx])
        f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
        t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
        data_adain = data_adain[:, t_idx]
        return tools.skeleton_adain_bone_length(data_numpy, data_adain)

    def apply_data_type(self, data_numpy):
        """Transforms data based on the specified type."""
        transformations = {
            'b': lambda x: tools.joint2bone()(x),
            'jm': tools.to_motion,
            'bm': lambda x: tools.to_motion(tools.joint2bone()(x))
        }
        return transformations.get(self.data_type, lambda x: x)(data_numpy)

    def top_k(self, score, top_k):
        """Calculates top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

