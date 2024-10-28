import torch
import numpy as np

mf_j_B = torch.from_numpy(np.load('./output/score/mf_j_B.npy'))
mf_b_B = torch.from_numpy(np.load('./output/score/mf_b_B.npy'))
# mf_jm_B = torch.from_numpy(np.load('./output/score/mf_jm_B.npy'))
# mf_bm_B = torch.from_numpy(np.load('./output/score/mf_bm_B.npy'))
ctr_j_B = torch.from_numpy(np.load('./output/score/ctr_j_B.npy'))
ctr_b_B = torch.from_numpy(np.load('./output/score/ctr_b_B.npy'))
# ctr_jm_B = torch.from_numpy(np.load('./output/score/ctr_jm_B.npy'))
# ctr_bm_B = torch.from_numpy(np.load('./output/score/ctr_bm_B.npy'))
teg_j_B = torch.from_numpy(np.load('./output/score/teg_j_B.npy'))
# teg_b_B = torch.from_numpy(np.load('./output/score/teg_b_B.npy'))
# teg_jm_B = torch.from_numpy(np.load('./output/score/teg_jm_B.npy'))
# te_bm_B = torch.from_numpy(np.load('./output/score/teg_bm_B.npy'))

if __name__ == '__main__':
    # scores = [mf_j_B, mf_b_B, mf_jm_B, mf_bm_B, ctr_j_B, ctr_b_B,
    #           ctr_jm_B, ctr_bm_B, teg_j_B, teg_b_B, teg_jm_B, te_bm_B]
    # final_score = torch.zeros_like(mf_j_B)
    # rate = [1.2, 1.2, 0.2, 0.2, 0.7826394637785445, 1.2, 0.6233394565637804, 0.2, 0.2, 1.2, 0.2, 0.2]
    # for idx, score in enumerate(scores):
    #     final_score += rate[idx] * score
    # np.save('./output/score/final_score.npy', final_score)
    scores = [mf_j_B, mf_b_B, ctr_j_B, ctr_b_B, teg_j_B]
    final_score = torch.zeros_like(mf_j_B)
    rate =  [1.5, 1.2550737832897505, 1.5, 0.9580145240671708, 1.2610920473728513]
    for idx, score in enumerate(scores):
        final_score += rate[idx] * score
    np.save('./output/score/final_score.npy', final_score)

