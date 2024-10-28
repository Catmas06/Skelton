import generate_score
import torch
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

# generate_score.generate_scores('./config/mf_j_2.yaml')
# generate_score.generate_scores('./config/mf_b_2.yaml')
# generate_score.generate_scores('./config/ctr_j_2.yaml')
# generate_score.generate_scores('./config/ctr_b_2.yaml')


train_A_label = torch.from_numpy(np.load('./data/train/train_label.npy'))
test_A_label = torch.from_numpy(np.load('./data/train/test_label.npy'))

mf_j_A = torch.from_numpy(np.load('./output/score/mf_j_A.npy'))
mf_b_A = torch.from_numpy(np.load('./output/score/mf_b_A.npy'))
mf_jm_A = torch.from_numpy(np.load('./output/score/mf_jm_A.npy'))
mf_bm_A = torch.from_numpy(np.load('./output/score/mf_bm_A.npy'))
ctr_j_A= torch.from_numpy(np.load('./output/score/ctr_j_A.npy'))
ctr_b_A= torch.from_numpy(np.load('./output/score/ctr_b_A.npy'))
ctr_jm_A= torch.from_numpy(np.load('./output/score/ctr_jm_A.npy'))
ctr_bm_A= torch.from_numpy(np.load('./output/score/ctr_bm_A.npy'))
teg_j_A= torch.from_numpy(np.load('./output/score/teg_j_A.npy'))
# teg_b_A= torch.from_numpy(np.load('./output/score/teg_b_A.npy'))
# teg_jm_A= torch.from_numpy(np.load('./output/score/teg_jm_A.npy'))
# teg_bm_A= torch.from_numpy(np.load('./output/score/teg_bm_A.npy'))

mf_j_test = torch.from_numpy(np.load('./output/score/mf_j_test.npy'))
mf_b_test = torch.from_numpy(np.load('./output/score/mf_b_test.npy'))
mf_jm_test = torch.from_numpy(np.load('./output/score/mf_jm_test.npy'))
mf_bm_test = torch.from_numpy(np.load('./output/score/mf_bm_test.npy'))
ctr_j_test= torch.from_numpy(np.load('./output/score/ctr_j_test.npy'))
ctr_b_test= torch.from_numpy(np.load('./output/score/ctr_b_test.npy'))
ctr_jm_test= torch.from_numpy(np.load('./output/score/ctr_jm_test.npy'))
ctr_bm_test= torch.from_numpy(np.load('./output/score/ctr_bm_test.npy'))
teg_j_test= torch.from_numpy(np.load('./output/score/teg_j_test.npy'))
# teg_b_test= torch.from_numpy(np.load('./output/score/teg_b_test.npy'))
# teg_jm_test= torch.from_numpy(np.load('./output/score/teg_jm_test.npy'))
# teg_bm_test= torch.from_numpy(np.load('./output/score/teg_bm_test.npy'))

mf_jm2_A = torch.from_numpy(np.load('./output/score/2/mf_jm_A.npy'))
mf_bm2_A = torch.from_numpy(np.load('./output/score/2/mf_bm_A.npy'))
ctr_jm2_A= torch.from_numpy(np.load('./output/score/2/ctr_jm_A.npy'))
ctr_bm2_A= torch.from_numpy(np.load('./output/score/2/ctr_bm_A.npy'))

mf_jm2_test = torch.from_numpy(np.load('./output/score/2/mf_jm_test.npy'))
mf_bm2_test = torch.from_numpy(np.load('./output/score/2/mf_bm_test.npy'))
ctr_jm2_test= torch.from_numpy(np.load('./output/score/2/ctr_jm_test.npy'))
ctr_bm2_test= torch.from_numpy(np.load('./output/score/2/ctr_bm_test.npy'))

def get_acc(score, label):
    # 找到最大预测值和对应的标签
    value, score_label = torch.max(score.data, 1)
    # to_acc = torch.mean((score_label == train_Alabel.data).float()).item()

    indices = []
    acc = [0] * 155
    # 计算每个样本的准确率
    for i in range(155):
        indices.append((label == i).nonzero(as_tuple=True)[0])
    # 计算准确率
    now_acc = []
    for j in range(155):
        for i in indices[j]:
            # 计算该样本的准确率
            if score_label[i] == label[i]:  # 对比整个目标标签
                acc[j] += 1
        now_acc.append(acc[j]/len(indices[j]))
    return now_acc


def get_weights(scores1, scores2, label):
    assert len(scores1) == len(scores2), f'len(scores1):{len(scores1)} != len(scores2):{len(scores2)}'
    weights = []
    for idx in range(len(scores1)):
        weights.append(torch.Tensor(get_acc(scores1[idx], label)))
        weights.append(torch.Tensor(get_acc(scores2[idx], label)))
    return weights


def objective(weights, ):
    right_num = total_num = 0
    for i in tqdm(range(len(test_A_label))):
        l = test_A_label[i]
        r_11 = mf_j_test [i]
        r_22 = mf_b_test [i]
        r_33 = mf_jm_test[i]
        r_44 = mf_bm_test[i]
        r_55 = ctr_j_test[i]
        r_66 = ctr_b_test[i]
        r_77 = ctr_jm_test[i]
        r_88 = ctr_bm_test[i]
        r_99 = teg_j_test[i]
        # r_1010 = teg_b_test[i]
        # r_1111 = teg_jm_test[i]
        # r_1212 = teg_bm_test[i]
        # r_99 = mf_j_test_new[i]
        # r_1010 = mf_b_test_new[i]
        # r_1111 = ctr_j_test_new[i]
        # r_1212 = ctr_b_test_new[i]

        # r = r_11 * weights[0] \
        #     + r_22 * weights[1] \
        #     + r_33 * weights[2] \
        #     + r_44 * weights[3] \
        #     + r_55 * weights[4] \
        #     + r_66 * weights[5] \
        #     + r_77 * weights[6] \
        #     + r_88 * weights[7] \
        #     + r_99 * weights[8] \
        #     + r_1010 * weights[9] \
        #     + r_1111 * weights[10] \
        #     + r_1212 * weights[11]

        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8]

        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

scores1 = [mf_jm_A, mf_bm_A, ctr_jm_A, ctr_bm_A]
scores2 = [mf_jm2_A, mf_bm2_A, ctr_jm2_A, ctr_bm2_A]
weights = get_weights(scores1, scores2, train_A_label)
# scores3 = [mf_jm_test, mf_bm_test, ctr_jm_test, ctr_bm_test]
# scores4 = [mf_jm2_test, mf_bm2_test, ctr_jm2_test, ctr_bm2_test]
# weights2 = get_weights(scores3, scores4, test_A_label)
# mf_j_test_new = (weights[0]*mf_j_test + weights[1]*mf_j2_test)
# mf_b_test_new = (weights[0]*mf_b_test + weights[1]*mf_b2_test)
# ctr_j_test_new = (weights[0]*ctr_j_test + weights[1]*ctr_j2_test)
# ctr_b_test_new = (weights[0]*ctr_b_test + weights[1]*ctr_b2_test)
mf_jm_test = (weights[0]*mf_jm_test + weights[1]*mf_jm2_test)/2
mf_bm_test = (weights[2]*mf_bm_test + weights[3]*mf_bm2_test)/2
ctr_jm_test = (weights[4]*ctr_jm_test + weights[5]*ctr_jm2_test)/2
ctr_bm_test = (weights[6]*ctr_bm_test + weights[7]*ctr_bm2_test)/2


def generate_B():
    # 生成增强融合jm、bm模型
    mf_jm_B = torch.from_numpy(np.load('./output/score/mf_jm_B.npy', allow_pickle=True))
    mf_bm_B = torch.from_numpy(np.load('./output/score/mf_bm_B.npy', allow_pickle=True))
    mf_jm2_B = torch.from_numpy(np.load('./output/score/2/mf_jm_B.npy', allow_pickle=True))
    mf_bm2_B = torch.from_numpy(np.load('./output/score/2/mf_bm_B.npy', allow_pickle=True))
    ctr_jm_B = torch.from_numpy(np.load('./output/score/ctr_jm_B.npy', allow_pickle=True))
    ctr_bm_B = torch.from_numpy(np.load('./output/score/ctr_bm_B.npy', allow_pickle=True))
    ctr_jm2_B = torch.from_numpy(np.load('./output/score/2/ctr_jm_B.npy', allow_pickle=True))
    ctr_bm2_B = torch.from_numpy(np.load('./output/score/2/ctr_bm_B.npy', allow_pickle=True))
    mf_jm_B = (weights[0]*mf_jm_B + weights[1]*mf_jm2_B)/2
    mf_bm_B = (weights[2]*mf_bm_B + weights[3]*mf_bm2_B)/2
    ctr_jm_B = (weights[4]*ctr_jm_B + weights[5]*ctr_jm2_B)/2
    ctr_bm_B = (weights[6]*ctr_bm_B + weights[7]*ctr_bm2_B)/2

    # 生成其他分数文件
    mf_j_B = torch.from_numpy(np.load('./output/score/mf_j_B.npy', allow_pickle=True))
    mf_b_B = torch.from_numpy(np.load('./output/score/mf_b_B.npy', allow_pickle=True))
    ctr_j_B = torch.from_numpy(np.load('./output/score/ctr_j_B.npy', allow_pickle=True))
    ctr_b_B = torch.from_numpy(np.load('./output/score/ctr_b_B.npy', allow_pickle=True))
    teg_j_B = torch.from_numpy(np.load('./output/score/teg_j_B.npy', allow_pickle=True))

    scores = [mf_j_B, mf_b_B, mf_jm_B, mf_bm_B, ctr_j_B, ctr_b_B,
              ctr_jm_B, ctr_bm_B, teg_j_B]
    final_score = torch.zeros_like(mf_j_B)
    rate = [1.2190926547336882, 1.01165679112073, 0.1, 0.1, 1.2082217291663677, 0.7448879840816925, 0.1, 0.1, 1.23540218370383]
    for idx, score in enumerate(scores):
        final_score += rate[idx] * score
    np.save('./output/score/final_score_2.npy', final_score)


if __name__ == '__main__':
    # space = [(0.0, 1.5) for i in range(9)]
    # result = gp_minimize(objective, space, n_calls=200, random_state=0)
    # print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    # print('Optimal weights: {}'.format(result.x))
    generate_B()
