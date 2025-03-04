import argparse
import os
from tqdm import tqdm
import multiprocessing
import numpy as np
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--use_mp', type=bool, default=False, help='use multi processing or not')
parser.add_argument('--modal', type=str, default='bone', help='use multi processing or not')

# uav graph
    # (10, 8), (8, 6), (9, 7), (7, 5), # arms
    # (15, 13), (13, 11), (16, 14), (14, 12), # legs
    # (11, 5), (12, 6), (11, 12), (5, 6), # torso
    # (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears

# switch set
sets = {'train', 'test_A'}

parts = {'joint', 'bone'}
graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

# bone
def gen_bone(set, set0, debug=False, is_master=True):
    if os.path.exists(f'./data/{set0}/{set}_bone.npy'):
        if is_master:
            print('bone modality already exists')
        return
    data = open_memmap(f'./data/{set0}/{set}_joint.npy',mode='r')
    if debug:
        data = data[:100,:]
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(f'./data/{set0}/{set}_bone.npy',dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    if debug:
        fp_sp = fp_sp[:100,:]
    for v1, v2 in tqdm(graph, desc='Generating bone modality'):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

# jmb
def merge_joint_bone_data(set, set0, debug=False, is_master=True):
    if os.path.exists(f'./data/{set0}/{set}_joint_bone.npy'):
        if is_master:
            print('joint_bone modality already exists')
        return
    data_jpt = open_memmap(f'./data/{set0}/{set}_joint.npy', mode='r')
    data_bone = open_memmap(f'./data/{set0}/{set}_bone.npy', mode='r')
    if debug:
        data_jpt = data_jpt[:100,:]
        data_bone = data_bone[:100,:]
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap(f'./data/{set0}/{set}_joint_bone.npy', dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone
