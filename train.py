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
from torch.utils.tensorboard import SummaryWriter


class Leaner():
    def __init__(self, arg):
        self.arg = arg
        self.dataloader = DataLoader(
            dataset=Feeder(**self.arg.train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )
        self.model = Model(graph=self.arg.model_args['graph'],
                           graph_args=self.arg.model_args['graph_args'])
        self.train_writer = SummaryWriter(os.path.join(arg.log_dir, 'train'), 'train')
        self.global_step = 0
        self.device = torch.device('cuda:{}'.format(self.arg.device))
        self.loss = torch.nn.CrossEntropyLoss()
        self.max_acc = 0.5

    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.log_dir):
            os.mkdir(self.arg.work_dir)
        with open('{}/log.txt'.format(self.arg.log_dir), 'a') as f:
            print(str, file=f)


    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, torch.nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'global_step': self.global_step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'max_acc': self.max_acc,
        }

    def save_to_checkpoint(self, state_dict, filename='weights'):
        if not os.path.exists(self.arg.model_saved_dir):
            os.mkdir(self.arg.model_saved_dir)
        link_name = f'{self.arg.model_saved_dir}/{filename}.pt'
        save_name = f'{self.arg.model_saved_dir}/temp_{filename}.pt'
        if os.path.exists(link_name):
            torch.save(state_dict, save_name)
            os.replace(save_name, link_name)
        else:
            torch.save(state_dict, link_name)

    def load_from_checkpoint(self):
        if not os.path.exists(self.arg.model_path):
            raise FileNotFoundError(f'path of checkpoint does not exist: {self.arg.model_path}')
        checkpoint = torch.load(self.arg.model_path, map_location=torch.device('cpu'))
        if hasattr(self.model, 'module') and isinstance(self.model.module, torch.nn.Module):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=float(self.arg.base_lr),
                                          weight_decay=self.arg.weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
        self.global_step = checkpoint['global_step']
        self.max_acc = checkpoint['max_acc']
        print(f'loaded checkpoint from {self.arg.model_path}')

    def train(self, epoch):
        if os.path.exists(self.arg.model_path):
            self.load_from_checkpoint()
        else:
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                              lr=float(self.arg.base_lr),
                                              weight_decay=self.arg.weight_decay)
        self.model.train()
        self.print_log(f'Training epoch: {epoch}')
        self.print_log(f'\t===== training from global steps {self.global_step} =====')
        self.epoch = epoch
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        mean_acc = 0
        for epoch in range(epoch):
            loss_value = []
            acc_value = []
            for data, label in tqdm(self.dataloader, desc='Training progress epoch {}'.format(epoch)):
                self.global_step += 1
                # data [N, 12, 300, 17, 2]
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                # label [N,]
                label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()

                # forward
                output = self.model(data)
                loss = self.loss(output, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 写入log
                loss_value.append(loss.data.item())
                value, predict_label = torch.max(output.data, 1)
                now_acc = torch.mean((predict_label == label.data).float()).item()
                acc_value.append(now_acc)
                self.train_writer.add_scalar('acc', now_acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

                # statistics
                self.lr = self.optimizer.param_groups[0]['lr']
                self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # save model after one epoch
            self.save_to_checkpoint(self.state_dict())

            # save the best model
            mean_acc = np.mean(acc_value)
            if mean_acc > self.max_acc:
                self.max_acc = mean_acc
                self.save_to_checkpoint(self.state_dict(), f'weights_acc_{self.max_acc:.4f}')
                self.save_to_checkpoint(self.state_dict(), f'best_weights')
            self.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}')
            self.print_log(f'\tMean training acc: {mean_acc:.4f}')
            self.print_log(f'\tMax training acc: {self.max_acc:.4f}')
            self.print_log(f'\t============ global steps {self.global_step} ============')

        self.save_to_checkpoint(self.state_dict(), f'last_weights_{mean_acc:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skelton-based Action Recognition')
    parser.add_argument('--config_dir',
                        default='./config/params.yaml',
                        help='path of the config file')
    parser.add_argument('--epoch')
    p = parser.parse_args()
    if p.config_dir is not None:
        with open(p.config_dir, 'r') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    leaner = Leaner(arg)
    leaner.train(arg.num_epoch)

