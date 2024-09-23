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
            # num_workers=self.arg.num_worker,
            num_workers=0,
            drop_last=True,
        )
        self.model = Model(graph=self.arg.model_args['graph'], graph_args=self.arg.model_args['graph_args'])
        self.train_writer = SummaryWriter(os.path.join(arg.log_dir, 'train'), 'train')
        self.global_step = 0
        self.device = torch.device('cuda:{}'.format(self.arg.device))
        self.loss = torch.nn.CrossEntropyLoss()

    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.log_dir):
            os.mkdir(self.arg.work_dir)
        with open('{}/log.txt'.format(self.arg.log_dir), 'a') as f:
            print(str, file=f)

    def save_to_checkpoint(self, filename='weights'):
        if not os.path.exists(self.arg.model_saved_dir):
            os.mkdir(self.arg.model_saved_dir)
        link_name = f'{self.arg.model_saved_dir}/{filename}.pt'
        save_name = f'{self.arg.model_saved_dir}/temp_{filename}.pt'
        if os.path.exists(link_name):
            torch.save(self.model.state_dict(), save_name)
            os.replace(save_name, link_name)
        else:
            torch.save(self.model.state_dict(), link_name)

    def train(self, epoch):
        self.model.to(self.device)
        self.model.train()
        print(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.arg.base_lr,
            weight_decay=self.arg.weight_decay)
        self.print_log('Training epoch: {}'.format(epoch))
        self.epoch = epoch
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        for epoch in range(epoch):
            for data, label in tqdm(self.dataloader, desc='Training progress epoch {}'.format(epoch)):
                self.global_step += 1
                # get data [N, 3, 300, 17, 2]
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                # data [N, 3, 128, 17, 1]
                # data = data[:,:,0:128,:,0:1].detach()
                label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()
                # data = data.clone().detach()
                # label = label.clone().detach()

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
                acc = torch.mean((predict_label == label.data).float())
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

                # statistics
                self.lr = self.optimizer.param_groups[0]['lr']
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
                # if self.global_step % self.arg.log_interval == 0:
                #     self.print_log(
                #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                #             batch_idx, len(loader), loss.data[0], lr))

                # statistics of time consumption and loss
                if self.global_step % 100 == 0:
                    self.save_to_checkpoint()

            self.print_log(
                '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
            torch.save(self.model.state_dict(), self.arg.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skelton-based Action Recognition')
    parser.add_argument('--config_dir',
                        default='./config/params.yaml',
                        help='path of the config file')
    parser.add_argument('--epoch')
    # load arg form config file
    p = parser.parse_args()
    if p.config_dir is not None:
        with open(p.config_dir, 'r') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    leaner = Leaner(arg)
    leaner.train(arg.num_epoch)

