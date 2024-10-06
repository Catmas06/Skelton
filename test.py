import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
from pre_data.feeder import Feeder
import pre_data.graph
from model.dmodel import Model
from torch.utils.data import DataLoader


class Val():
    def __init__(self, arg):
        self.arg = arg
        self.dataloader = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        self.model = Model(graph=self.arg.model_args['graph'],
                           graph_args=self.arg.model_args['graph_args'])
        self.device = torch.device('cuda:{}'.format(self.arg.test_device))
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.acc = 0
        self.loss = 100
        self.max_acc = 0.65

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
        print(f'loaded testing checkpoint from {path}')

    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.log_dir):
            os.mkdir(self.arg.work_dir)
        with open('{}/log.txt'.format(self.arg.log_dir), 'a') as f:
            print(str, file=f)

    def test(self, epoch, save_model=False):
        self.load_from_checkpoint()
        self.model.eval()
        global_acc = []
        loss_value = []
        with torch.no_grad():
            for data, label in tqdm(self.dataloader, desc='Testing progress epoch {}'.format(epoch)):
                # get data [N, 3, 300, 17, 2]
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()

                # forward
                output = self.model(data)
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
            self.print_log(f'\tMean testing loss: {self.loss:.4f}')
            self.print_log(f'\tMean testing  acc: {self.acc:.4f}')
            self.print_log(f'\t Max testing  acc: {self.max_acc:.4f}')


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
        with open(p.config_dir, 'r') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)
    parser.add_argument('--test_path',
                        help='path of the config file')

    arg = parser.parse_args()
    leaner = Val(arg)
    leaner.test()

