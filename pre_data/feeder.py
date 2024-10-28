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
                 window_size=-1, normalization=False, debug=False, use_mmap=False,
                 is_master=True, p_interval=[1], use_clean=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data = []
        self.label = []
        self.use_clean = use_clean
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.is_master = is_master
        self.load_data()
        self.p_interval = p_interval

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        data = None
        label = None
        if 'train' in os.path.split(self.data_path)[-1]:
            self.set = 'train'
        elif 'test' in os.path.split(self.data_path)[-1]:
            self.set = 'test'
        else:
            raise ValueError('The data_path must contain words train or test')
        if 'train' in os.path.split(self.data_path)[-2]:
            self.set0 = 'train'
        elif 'test' in os.path.split(self.data_path)[-2]:
            self.set0 = 'test'
        else:
            raise ValueError('The data_path must contain words train or test')
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist'
        if self.set0 == 'train':
            assert os.path.exists(self.label_path), f'{self.label_path} does not exist'
            label = np.load(self.label_path)
        # load data
        if self.use_mmap:
            data = np.load(self.data_path, mmap_mode='r')
        else:
            data = np.load(self.data_path)
        N, C, T, V, M = data.shape
        if not os.path.exists(f'./data/{self.set0}/{self.set}_joint_bone_motion.npy'):
            gen_modal.gen_bone(self.set, self.set0, debug=self.debug, is_master=self.is_master)
            gen_modal.merge_joint_bone_data(self.set, self.set0, debug=self.debug, is_master=self.is_master)
            motion = np.load(f'./data/{self.set0}/{self.set}_joint_bone.npy')
            data = np.array(motion)
            for t in tqdm(range(T - 1), desc='Generating motion modality'):
                motion[:, :, t, :, :] = motion[:, :, t + 1, :, :] - motion[:, :, t, :, :]
            motion[:, :, T - 1, :, :] = 0
            # C:[joint, bone, joint_motion, bone_motion] 4*3=12
            data = np.concatenate((data, motion), axis=1)
            np.save(f'./data/{self.set0}/{self.set}_joint_bone_motion.npy', data)
        else:
            data = np.load(f'./data/{self.set0}/{self.set}_joint_bone_motion.npy')
            if self.is_master:
                print('data already prepared')
        for index in range(len(data)):
            valid_frame_num = np.sum(data[index].sum(0).sum(-1).sum(-1) != 0)
            if valid_frame_num > 0 or self.set == 'test' or self.use_clean is not True:
                self.data.append(data[index])
                if label is not None:
                    self.label.append(label[index])
        self.data = np.stack(self.data, axis=0)
        if label is not None:
            self.label = np.stack(self.label, axis=0)
        if self.debug:
            if self.set0 == 'test':
                self.label = self.label[0:100]
            self.data = self.data[0:100]


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = -155
        if self.set0 == 'train':
            label = self.label[index]
        data_numpy = np.array(data_numpy)
        # random crop
        if self.window_size!=-1:
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
            if valid_frame_num == 0:
                # self.data = np.delete(self.data, index, axis=0)
                print(f'The data[{index}] is 0.')
                data_numpy = tools.valid_crop_resize(data_numpy, 300, self.p_interval, self.window_size)
            else:
                data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
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
    os.chdir('..')
    # feeder = Feeder('./data/train/test_joint.npy', './data/train/test_label.npy', debug=False)
    feeder = Feeder('./data/test/test_joint_bone_motion_B.npy', None, debug=False)
    it, label = feeder.__getitem__(14)
    print(it)
