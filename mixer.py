import torch
import numpy as np
import os
from utils import tools
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skopt import gp_minimize
from pre_data.graph import Graph
import model.dmodel as TEG
import model.ske_mixf as MF
import model.ctrgcn_xyz as CTR

# 假设 te_joint 是模型输出的张量
# te_joint = torch.from_numpy(np.random.rand(16432, 155))# 示例数据
train_Alabel = torch.from_numpy(np.load('./data/train/train_label.npy')) # 假设标签文件已经存在
test_Alabel = torch.from_numpy(np.load('./data/train/test_label.npy'))
mf_j_A = torch.from_numpy(np.load('./output/score/mf_j_A.npy'))
# mf_b_A = torch.from_numpy(np.load('./output/score/mf_b_A.npy'))
# mf_jm_A = torch.from_numpy(np.load('./output/score/mf_jm_A.npy'))
# mf_bm_A = torch.from_numpy(np.load('./output/score/mf_bm_A.npy'))
# ctr_j_A = torch.from_numpy(np.load('./output/score/ctr_j_A.npy'))
# ctr_b_A = torch.from_numpy(np.load('./output/score/ctr_b_A.npy'))
# ctr_jm_A = torch.from_numpy(np.load('./output/score/ctr_jm_A.npy'))
# cre_bm_A = torch.from_numpy(np.load('./output/score/ctr_bm_A.npy'))
# mf_j_test = torch.from_numpy(np.load('./output/score/test_mf_j_test.npy'))
# mf_b_test = torch.from_numpy(np.load('./output/score/test_mf_b_test.npy'))
# mf_jm_test = torch.from_numpy(np.load('./output/score/test_mf_jm_test.npy'))
# mf_bm_test = torch.from_numpy(np.load('./output/score/test_mf_bm_test.npy'))
# ctr_j_test= torch.from_numpy(np.load('./output/score/test_ctr_j_test.npy'))
# ctr_b_test= torch.from_numpy(np.load('./output/score/test_ctr_b_test.npy'))
# ctr_jm_test= torch.from_numpy(np.load('./output/score/test_ctr_jm_test.npy'))
# ctr_bm_test= torch.from_numpy(np.load('./output/score/test_ctr_bm_test.npy'))
# teg_j_test= torch.from_numpy(np.load('./output/score/test_teg_j_test.npy'))
# teg_b_test= torch.from_numpy(np.load('./output/score/test_teg_b_test.npy'))
# teg_jm_test= torch.from_numpy(np.load('./output/score/test_teg_jm_test.npy'))
# teg_bm_test= torch.from_numpy(np.load('./output/score/test_teg_bm_test.npy'))

def get_acc(te_joint, target):
    # 找到最大预测值和对应的标签
    value, te_joint_label = torch.max(te_joint.data, 1)
    # to_acc = torch.mean((te_joint_label == train_Alabel.data).float()).item()

    indices = []
    acc = [0] * 155
    # 计算每个样本的准确率
    for i in range(155):
        indices.append((target == i).nonzero(as_tuple=True)[0])
    # 计算准确率
    now_acc = []
    for j in range(155):
        for i in indices[j]:
            # 计算该样本的准确率
            if te_joint_label[i] == target[i]:  # 对比整个目标标签
                acc[j] += 1
        now_acc.append(acc[j]/len(indices[j]))
    return now_acc

def load_from_checkpoint(path, model, device='cuda:0'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'path of checkpoint does not exist: {path}')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if hasattr(model, 'module') and isinstance(model.module, torch.nn.Module):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(f'loaded testing checkpoint from {path}')

def demo():
    mf_j_acc = get_acc(mf_j_A, train_Alabel)
    mf_b_acc = get_acc(mf_b_A, train_Alabel)
    mf_jm_acc = get_acc(mf_jm_A, train_Alabel)
    mf_bm_acc = get_acc(mf_bm_A, train_Alabel)
    ctr_j_acc = get_acc(ctr_j_A, train_Alabel)
    ctr_b_acc = get_acc(ctr_b_A, train_Alabel)
    ctr_jm_acc = get_acc(ctr_jm_A, train_Alabel)
    cre_bm_acc = get_acc(cre_bm_A, train_Alabel)
    weight = np.stack(np.array([mf_j_acc,mf_b_acc,mf_jm_acc,mf_bm_acc,ctr_j_acc,ctr_b_acc,ctr_jm_acc,cre_bm_acc]), axis=0)
    weight = torch.from_numpy(weight)
    data = np.stack((mf_j_test.numpy(), mf_b_test.numpy(),mf_jm_test.numpy(),mf_bm_test.numpy(),ctr_j_test.numpy(),ctr_b_test.numpy(),ctr_jm_test.numpy(),ctr_bm_test.numpy()), axis=1)
    data_tensor = torch.from_numpy(data)
    weighted_result = []
    for i in range(2000):
        result = torch.mul(data_tensor[i], weight)  # 去掉 detach() 和 numpy()
        weighted_result.append(result)
    output = torch.sum(torch.stack(weighted_result), dim=1)
    value, out_label = torch.max(output.data, 1)

    now_acc = torch.mean((out_label == test_Alabel.data).float()).item()
    print(now_acc)
    confidence = np.array(np.array(output))


def objective(weights, ):
    right_num = total_num = 0
    for i in tqdm(range(len(test_Alabel))):
        l = test_Alabel[i]
        r_11 = mf_j_test [i]
        r_22 = mf_b_test [i]
        r_33 = mf_jm_test[i]
        r_44 = mf_bm_test[i]
        r_55 = ctr_j_test[i]
        r_66 = ctr_b_test[i]
        r_77 = ctr_jm_test[i]
        r_88 = ctr_bm_test[i]
        r_99 = teg_j_test[i]
        r_1010 = teg_b_test[i]
        r_1111 = teg_jm_test[i]
        r_1212 = teg_bm_test[i]

        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r_1010 * weights[9] \
            + r_1111 * weights[10] \
            + r_1212 * weights[11]

        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

if __name__ == '__main__':
    # space = [(0.2, 1.2) for i in range(12)]
    # result = gp_minimize(objective, space, n_calls=200, random_state=0)
    # print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    # print('Optimal weights: {}'.format(result.x))
    mf_j_acc = get_acc(mf_j_A, train_Alabel)
    # mf_b_acc = get_acc(mf_b_A, train_Alabel)
    # mf_jm_acc = get_acc(mf_jm_A, train_Alabel)
    # mf_bm_acc = get_acc(mf_bm_A, train_Alabel)
    # ctr_j_acc = get_acc(ctr_j_A, train_Alabel)
    # ctr_b_acc = get_acc(ctr_b_A, train_Alabel)
    # ctr_jm_acc = get_acc(ctr_jm_A, train_Alabel)
    # ctr_bm_acc = get_acc(ctr_bm_A, train_Alabel)
    # teg_j_acc = get_acc(teg_j_A, train_Alabel)
    # teg_b_acc = get_acc(teg_b_A, train_Alabel)
    # teg_jm_acc = get_acc(teg_jm_A, train_Alabel)
    # teg_bm_acc = get_acc(teg_bm_A, train_Alabel)



