import torch
import numpy as np
import yaml
import os
import pre_data.graph as graph
import model.ske_mixf as MF
import model.ctrgcn_xyz as CTR
import model.dmodel as TEG
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pre_data.feeder import Feeder

def load_from_checkpoint(path, model, device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'path of checkpoint does not exist: {path}')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if hasattr(model, 'module') and isinstance(model.module, torch.nn.Module):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(f'loaded testing checkpoint from {path}')

def generate_score(model_type, model_path, data_idx, data_path,
                   label_path=None, save_path=None, device='cuda:0',
                   window_size=64, batch_size=512, p_interval=[0.95]):
    assert os.path.exists(data_path), f'{data_path} is invalid'
    model = None
    if model_type == 'MF':
        model = MF.Model(graph=graph.Graph())
    elif model_type == 'CTR':
        model = CTR.Model(graph=graph.Graph())
    elif model_type == 'TEG':
        model = TEG.Model(graph=graph.Graph())
    else:
        raise ValueError(f'The model_type is not supported: {model_type}')
    load_from_checkpoint(model_path, model, device)
    model.eval()
    dataloader = DataLoader(
        dataset=Feeder(data_path=data_path,
                       label_path=label_path,
                       window_size=window_size,
                       p_interval=p_interval,
                       is_master=False,
                       use_clean=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    confidence = None
    with torch.no_grad():
        for data, label in tqdm(dataloader,
                                desc=f'Generating score for {model_type}, in {data_path}:{data_idx}'):
            data = data[:, data_idx:data_idx + 3, :].to(device)
            output = model(data)
            if confidence is None:
                confidence = np.array(np.array(output.cpu()))
            else:
                confidence = np.append(confidence, np.array(output.cpu()), axis=0)
    np.save(save_path, confidence)

def generate_scores(config_path):
    arg = None
    with open(config_path, 'rb') as f:
        arg = yaml.safe_load(f)
    arg['train_feeder_args']['p_interval'] = [0.95]
    # generate A train score
    # generate_score(model_type=arg['model_type'],
    #                model_path=f"{arg['model_saved_dir']}/best_test_weights.pt",
    #                data_idx=arg['data_idx'],
    #                data_path=arg['train_feeder_args']['data_path'],
    #                label_path=arg['train_feeder_args']['label_path'],
    #                save_path=arg['confidence_A_path'],
    #                device=arg['score_device'],
    #                window_size=arg['train_feeder_args']['window_size'],
    #                batch_size=arg['score_batch_size'],
    #                p_interval=arg['train_feeder_args']['p_interval'])
    # # generate A test score
    # generate_score(model_type=arg['model_type'],
    #                model_path=f"{arg['model_saved_dir']}/best_test_weights.pt",
    #                data_idx=arg['data_idx'],
    #                data_path=arg['test_feeder_args']['data_path'],
    #                label_path=arg['test_feeder_args']['label_path'],
    #                save_path=arg['confidence_test_path'],
    #                device=arg['score_device'],
    #                window_size=arg['test_feeder_args']['window_size'],
    #                batch_size=arg['score_batch_size'],
    #                p_interval=arg['test_feeder_args']['p_interval'])
    # generate B test score
    generate_score(model_type=arg['model_type'],
                   model_path=f"{arg['model_saved_dir']}/best_test_weights.pt",
                   data_idx=arg['data_idx'],
                   data_path='./data/test/test_joint.npy',
                   label_path=None,
                   save_path=arg['confidence_B_path'],
                   device=arg['score_device'],
                   window_size=arg['test_feeder_args']['window_size'],
                   batch_size=arg['score_batch_size'],
                   p_interval=arg['test_feeder_args']['p_interval'])


if __name__ == '__main__':
    generate_scores('./config/mf_j.yaml')
    generate_scores('./config/mf_b.yaml')
    generate_scores('./config/mf_jm.yaml')
    generate_scores('./config/mf_bm.yaml')
    generate_scores('./config/ctr_j.yaml')
    generate_scores('./config/ctr_b.yaml')
    generate_scores('./config/ctr_jm.yaml')
    generate_scores('./config/ctr_bm.yaml')
    generate_scores('./config/teg_j.yaml')
    # generate_scores('./config/teg_b.yaml')
    # generate_scores('./config/teg_jm.yaml')
    # generate_scores('./config/teg_bm.yaml')
