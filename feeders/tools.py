import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeederUtils:
    @staticmethod
    def crop_resize(data, valid_frames, p_interval, window, threshold):
        C, T, V, M = data.shape
        valid_size = valid_frames
        if len(p_interval) == 1:
            p = p_interval[0]
            cropped_length = max(int(valid_size * p), threshold)
            bias = int((1 - p) * valid_size / 2)
            data = data[:, bias:bias + cropped_length, :, :]
        else:
            p = np.random.uniform(p_interval[0], p_interval[1])
            cropped_length = max(int(valid_size * p), threshold)
            bias = np.random.randint(0, valid_size - cropped_length + 1)
            data = data[:, bias:bias + cropped_length, :, :]
        
        data = torch.tensor(data, dtype=torch.float)
        data = F.interpolate(data.view(V * M, C, cropped_length).permute(1, 2, 0).unsqueeze(0), size=window, mode='linear', align_corners=False)
        data = data.squeeze(0).permute(2, 0, 1).contiguous().view(C, window, V, M).numpy()
        index_t = 2 * torch.linspace(bias, bias + cropped_length, window) / valid_size - 1
        return data, index_t.numpy()

    @staticmethod
    def crop_uniform(data, valid_frames, p_interval, window, threshold):
        C, T, V, M = data.shape
        valid_size = valid_frames
        p = np.random.uniform(p_interval[0], p_interval[1])
        cropped_length = max(int(valid_size * p), threshold)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        indices = np.linspace(bias, bias + cropped_length - 1, window, dtype=int)
        data = data[:, indices, :, :]
        
        data = torch.tensor(data, dtype=torch.float)
        data = F.interpolate(data.view(V * M, C, len(indices)).permute(1, 2, 0).unsqueeze(0), size=window, mode='linear', align_corners=False)
        data = data.squeeze(0).permute(2, 0, 1).contiguous().view(C, window, V, M).numpy()
        index_t = 2 * torch.tensor(indices) / valid_size - 1
        return data, index_t.numpy()

class DataAugmentation:
    @staticmethod
    def scale(data, scale=0.2, prob=0.5):
        return data * (1 + np.random.uniform(-scale, scale, size=(3, 1, 1, 1))) if random.random() < prob else data

    @staticmethod
    def spatial_flip(data, prob=0.5):
        uav_order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        return data[:, :, uav_order, :] if random.random() < prob else data

    @staticmethod
    def rotate(data, axis=None, angle=None, prob=0.5):
        if random.random() >= prob:
            return data
        axis = axis or random.randint(0, 2)
        angle = angle or random.uniform(-30, 30)
        rotation_matrix = DataAugmentation._rotation_matrix(axis, math.radians(angle))
        return np.dot(data.transpose(1, 2, 3, 0), rotation_matrix).transpose(3, 0, 1, 2)

    @staticmethod
    def _rotation_matrix(axis, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        if axis == 0:
            return np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        elif axis == 1:
            return np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        elif axis == 2:
            return np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

    @staticmethod
    def gaussian_noise(data, mean=0, std=0.05, prob=0.5):
        return data + np.random.normal(mean, std, size=data.shape) if random.random() < prob else data

class JointBoneConversion(nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def forward(self, joint_data):
        bone_data = np.zeros_like(joint_data)
        for v1, v2 in self.pairs:
            bone_data[:, :, v1, :] = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
        return bone_data

class SkeletonAdaIN:
    @staticmethod
    def apply(input_data, reference_data, eps=1e-5):
        bone_i = JointBoneConversion(joint_pairs).forward(input_data)
        bone_r = JointBoneConversion(joint_pairs).forward(reference_data)
        length_scale = (np.linalg.norm(bone_r, axis=0) + eps) / (np.linalg.norm(bone_i, axis=0) + eps)
        return BoneToJointConverter().apply(bone_i * length_scale)

class BoneToJointConverter:
    def apply(self, bone_data, center_data):
        joint_data = np.zeros_like(bone_data)
        joint_data[:, :, self.center, :] = center_data
        for pairs in self.pair_groups:
            for v1, v2 in pairs:
                joint_data[:, :, v1, :] += bone_data[:, :, v1, :] + joint_data[:, :, v2, :]
        return joint_data

class GaussianBlur(nn.Module):
    def __init__(self, channels=3, kernel_size=15, sigma_range=[0.1, 2], prob=0.5):
        super().__init__()
        self.channels, self.kernel_size, self.prob = channels, kernel_size, prob
        radius = kernel_size // 2
        self.kernel_index = np.arange(-radius, radius + 1)
        self.sigma_range = sigma_range

    def forward(self, x):
        sigma = random.uniform(*self.sigma_range)
        kernel = np.exp(-self.kernel_index ** 2 / (2.0 * sigma ** 2))
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).repeat(self.channels, 1, 1, 1)
        kernel = kernel / kernel.sum()
        if random.random() < self.prob:
            x = F.conv2d(x.permute(3, 0, 2, 1), kernel, padding=(0, self.kernel_size // 2), groups=self.channels)
        return x.permute(1, -1, -2, 0).numpy()

def to_motion(input_data):
    motion_data = np.diff(input_data, axis=1, prepend=np.zeros_like(input_data[:, :1]))
    return motion_data
