import numpy as np
import os
from torch.utils.data import Dataset
from numpy.lib.format import open_memmap
from tqdm import tqdm
from utils import tools
from pre_data import gen_modal


class Feeder(Dataset):
    """Dataset of training data, generate from .npy files
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence

        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
    """
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 is_master=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.is_master = is_master
        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if 'train' in self.data_path:
            self.set = 'train'
        elif 'test' in self.data_path:
            self.set = 'test'
        else:
            raise ValueError('The data_path must contain words train or test')
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist'
        assert os.path.exists(self.label_path), f'{self.label_path} does not exist'
        self.label = np.load(self.label_path)
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        N, C, T, V, M = self.data.shape
        gen_modal.gen_bone(self.set, debug=self.debug, is_master=self.is_master)
        gen_modal.merge_joint_bone_data(self.set, debug=self.debug, is_master=self.is_master)
        if not os.path.exists(f'./data/{self.set}_joint_bone_motion.npy'):
            motion = np.load(f'./data/{self.set}_joint_bone.npy')
            self.data = np.array(motion)
            for t in tqdm(range(T - 1), desc='Generating motion modality'):
                motion[:, :, t, :, :] = motion[:, :, t + 1, :, :] - motion[:, :, t, :, :]
            motion[:, :, T - 1, :, :] = 0
            # C:[joint, bone, joint_motion, bone_motion] 4*3=12
            self.data = np.concatenate((self.data, motion), axis=1)
            np.save(f'./data/{self.set}_joint_bone_motion.npy', self.data)
        else:
            self.data = np.load(f'./data/{self.set}_joint_bone_motion.npy')
            if self.is_master:
                print('data already prepared')
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


if __name__ == '__main__':
    feeder = Feeder('./data/train_joint.npy', './data/train_label.npy', debug=False)
    it, label = feeder.__getitem__(14)
    print(it)
