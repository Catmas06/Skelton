import torch
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

test_Alabel = torch.from_numpy(np.load('./data/train/test_label.npy'))
mf_j_test = torch.from_numpy(np.load('./output/score/test_mf_j_test.npy'))
mf_b_test = torch.from_numpy(np.load('./output/score/test_mf_b_test.npy'))
mf_jm_test = torch.from_numpy(np.load('./output/score/test_mf_jm_test.npy'))
mf_bm_test = torch.from_numpy(np.load('./output/score/test_mf_bm_test.npy'))
ctr_j_test= torch.from_numpy(np.load('./output/score/test_ctr_j_test.npy'))
ctr_b_test= torch.from_numpy(np.load('./output/score/test_ctr_b_test.npy'))
ctr_jm_test= torch.from_numpy(np.load('./output/score/test_ctr_jm_test.npy'))
ctr_bm_test= torch.from_numpy(np.load('./output/score/test_ctr_bm_test.npy'))
teg_j_test= torch.from_numpy(np.load('./output/score/test_teg_j_test.npy'))
teg_b_test= torch.from_numpy(np.load('./output/score/test_teg_b_test.npy'))
teg_jm_test= torch.from_numpy(np.load('./output/score/test_teg_jm_test.npy'))
teg_bm_test= torch.from_numpy(np.load('./output/score/test_teg_bm_test.npy'))

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

def mix_2(score1, weight1, score2, weight2):
    from generate_score import generate_scores

    return score1*weight1 + score2*weight2

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
    space = [(0.2, 1.2) for i in range(12)]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))



