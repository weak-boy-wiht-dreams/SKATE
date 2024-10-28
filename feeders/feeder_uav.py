import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools

# 定义 Feeder 数据集类，继承自 PyTorch 的 Dataset 类
class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):
        """
        :param data_path: 数据文件路径
        :param label_path: 标签文件路径
        :param p_interval: 数据片段的间隔
        :param split: 数据集划分（train 或 test）
        :param data_type: 数据类型
        :param aug_method: 数据增强方法
        :param intra_p: 同一实例数据增强的概率
        :param inter_p: 不同实例数据增强的概率
        :param window_size: 窗口大小
        :param debug: 是否启用调试模式
        :param thres: 阈值
        :param uniform: 是否均匀裁剪
        :param partition: 是否进行分区
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()  # 加载数据

    def load_data(self):
        """加载数据和标签，根据 split 的值决定加载训练集还是测试集"""
        if self.split == 'train':
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

    def __len__(self):
        """返回数据集的长度"""
        return len(self.label)

    def __iter__(self):
        """定义迭代方法"""
        return self

    def __getitem__(self, index):
        """获取指定索引的数据样本和标签，并进行必要的处理和数据增强"""
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)  # 将数据转为 numpy 数组格式
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)  # 计算有效帧数
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)  # 计算人数
        if valid_frame_num == 0:  # 如果有效帧数为 0，则返回空数据
            return np.zeros((3, self.window_size, 17, 2)), np.zeros((self.window_size,)), label, index

        # 裁剪或调整数据大小
        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                           self.window_size, self.thres)
        else:
            data_numpy, index_t = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        # 数据增强（仅在训练集上进行）
        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)  # 随机生成概率
            if p < self.intra_p:
                # 各种数据增强方法
                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method and num_people == 2:
                    if np.random.rand(1) < 0.5:
                        axis_next = np.random.randint(0, 1)
                        temp = data_numpy.copy()
                        C, T, V, M = data_numpy.shape
                        x_new = np.zeros((C, T, V))
                        temp[:, :, :, axis_next] = x_new
                        data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = np.array(self.data[adain_idx])
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)
            else:
                data_numpy = data_numpy.copy()

        # 根据数据类型处理模态
        if self.data_type == 'b':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        """返回命中 top_k 的准确率"""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


# 动态导入模块的函数
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
