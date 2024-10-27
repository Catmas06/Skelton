import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
from pre_data.feeder import Feeder
import pre_data.graph as graph
import model.ske_mixf as MF
import model.ctrgcn_xyz as CTR
import model.dmodel as TEG
from utils import tools
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class NumpyDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = tools.valid_crop_resize(sample, 300, [0.95], 64)
        return torch.tensor(sample, dtype=torch.float32)


class Val():
    def __init__(self, arg):
        self.arg = arg
        self.dataloader = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args, is_master=False),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        self.model_type = arg.model_type
        if self.model_type == 'MF':
            self.model = MF.Model(graph=graph.Graph())
        elif self.model_type == 'CTR':
            self.model = CTR.Model(graph=graph.Graph())
        elif self.model_type == 'TEG':
            self.model = TEG.Model(graph=graph.Graph())
        else:
            raise ValueError(f'The model_type is not supported: {self.model_type}')
        self.device = torch.device('cuda:{}'.format(self.arg.test_device))
        self.data_idx = arg.data_idx
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.test_writer = SummaryWriter(os.path.join(arg.log_dir, 'test'), 'test')
        self.global_epoch = 0
        self.acc = 0
        self.loss = 100
        self.max_acc = 0.3

    def load_from_checkpoint(self):
        path = self.arg.test_path
        if not os.path.exists(path):
            raise FileNotFoundError(f'path of checkpoint does not exist: {path}')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if hasattr(self.model, 'module') and isinstance(self.model.module, torch.nn.Module):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.global_epoch = checkpoint['global_epoch']
        print(f'loaded testing checkpoint from {path}')

    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.log_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/log.txt'.format(self.arg.log_dir), 'a') as f:
            print(str, file=f)

    def test(self, epoch=0):
        self.load_from_checkpoint()
        self.model.eval()
        global_acc = []
        loss_value = []
        confidence = None
        with torch.no_grad():
            for data, label in tqdm(self.dataloader, desc='Testing progress epoch {}'.format(epoch)):
                # get data [N, 3, 300, 17, 2]
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                data = data[:,self.data_idx:self.data_idx+3,:]
                label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()

                # forward
                output = self.model(data)
                if confidence is None:
                    confidence = np.array(np.array(output.cpu()))
                else:
                    confidence = np.append(confidence, np.array(output.cpu()), axis=0)
                loss = self.loss_func(output, label)
                # 写入log
                loss_value.append(loss.data.item())
                value, predict_label = torch.max(output.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                global_acc.append(acc.item())
            self.acc = np.mean(global_acc)
            self.loss = np.mean(loss_value)
            if self.acc > self.max_acc:
                self.max_acc = self.acc
            self.test_writer.add_scalar('acc', self.acc, self.global_epoch)
            self.test_writer.add_scalar('loss', self.loss, self.global_epoch)
            self.print_log(f'\tMean  testing loss: {self.loss:.4f}')
            self.print_log(f'\tMean  testing  acc: {self.acc:.4f}')
            self.print_log(f'\t Max  testing  acc: {self.max_acc:.4f}')
        # np.save(os.path.join(self.arg.confidence_file_path), confidence)

    def last_test(self, path, confidence_file_path=None):
        self.load_from_checkpoint()
        self.model.eval()
        confidence = None
        # dataset = NumpyDataset(path)
        # loader = DataLoader(
        #     dataset=dataset,
        #     batch_size=64,
        #     shuffle=False,
        #     num_workers=0,
        #     drop_last=False,
        # )
        with torch.no_grad():
            for data in tqdm(self.dataloader, desc='Testing progress'):
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                data = data[:,3:6,:]
                output = self.model(data)
                if confidence is None:
                    confidence = np.array(np.array(output.cpu()))
                else:
                    confidence = np.append(confidence, np.array(output.cpu()), axis=0)
        if confidence_file_path is None:
            np.save(os.path.join(self.arg.confidence_file_path), confidence)
        else:
            np.save(confidence_file_path, confidence)
        return confidence

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Skelton-based Action Recognition')
    parser.add_argument('--config_dir',
                        default='./config/params.yaml',
                        help='path of the config file')
    # load arg form config file
    p = parser.parse_args()
    if p.config_dir is not None:
        with open(p.config_dir, 'r', encoding='utf-8') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)
    parser.add_argument('--test_path',
                        help='path of the config file')

    arg = parser.parse_args()
    leaner = Val(arg)
    leaner.last_test('./data/test/test_joint_B.npy')
    # leaner.test()

