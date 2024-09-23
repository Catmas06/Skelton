import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
from pre_data.feeder import Feeder
from model.dmodel import Model
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter


class Val():
    def __init__(self, arg):
        self.arg = arg
        self.dataloader = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=False,
            # num_workers=self.arg.num_worker,
            num_workers=0,
            drop_last=False,
        )
        self.model = Model(graph=self.arg.model_args['graph'], graph_args=self.arg.model_args['graph_args'])
        self.test_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.device = torch.device('cuda:{}'.format(self.arg.device))
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = None


    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.work_dir):
            os.mkdir(self.arg.work_dir)
        with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
            print(str, file=f)

    def test(self, epoch, save_model=False):
        if os.path.exists('./runs/uav/weights.pt'):
            self.model.load_state_dict(torch.load('./runs/uav/weights.pt') )
        else:
            print('错误\n')
        self.model = Model().to(self.device)
        self.model.eval()
        print(self.model)
        global_acc = 0
        for data, label in tqdm(self.dataloader, desc='Test progress epoch {}'.format(epoch)):
            self.global_step += 1
            # get data [N, 3, 300, 17, 2]
            data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
            # data [N, 3, 128, 17, 1]
            data = data[:,:,0:128,:,0:1].detach()
            label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()
            # data = data.clone().detach()
            # label = label.clone().detach()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # 写入log
            loss_value = []
            loss_value.append(loss.data.item())
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc /= len(self.dataloader)
            global_acc += acc
            self.test_writer.add_scalar('acc', acc, self.global_step)
            self.test_writer.add_scalar('loss', loss.data.item(), self.global_step)
        self.print_log('\tMean testing acc: {:.4f}.'.format(global_acc))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Skelton-based Action Recognition')
    parser.add_argument('--config_dir',
                        default='./config/uav/train.yaml',
                        help='path of the config file')
    # load arg form config file
    p = parser.parse_args()
    if p.config_dir is not None:
        with open(p.config_dir, 'r') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    leaner = Val(arg)
    leaner.test(arg.num_epoch)

