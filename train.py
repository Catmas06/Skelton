import argparse
import yaml
import os
import torch
import numpy as np
import test
import random
from tqdm import tqdm
from pre_data.feeder import Feeder
import pre_data.graph as graph
import model.ske_mixf as MF
import model.ctrgcn_xyz as CTR
import model.dmodel as TEG
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def setup_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

def print_log(log_dir, str):
    print(str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open('{}/log.txt'.format(log_dir), 'a') as f:
        print(str, file=f)


class Leaner():
    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.global_epoch = 0
        self.model_type = arg.model_type
        self.data_idx = arg.data_idx
        self.device = torch.device('cuda:{}'.format(self.arg.device))
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = self.arg.base_lr
        self.max_test_acc = 0.3
        self.max_acc = 0.87
        self.tester = test.Val(arg)

    def print_log(self, str):
        print(str)
        if not os.path.exists(self.arg.log_dir):
            os.makedirs(self.arg.log_dir)
        with open('{}/log.txt'.format(self.arg.log_dir), 'a') as f:
            print(str, file=f)


    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, torch.nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'max_acc': self.max_acc,
            'max_test_acc': self.max_test_acc
        }

    def load_optimizer(self):
        if self.arg.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                              lr=float(self.arg.base_lr),
                                              weight_decay=float(self.arg.weight_decay))
        elif self.arg.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=float(self.arg.base_lr),
                                             momentum=0.9,
                                             nesterov=self.arg.nesterov,
                                             weight_decay=float(self.arg.weight_decay))
        else:
            raise ValueError('Unknown optimizer')

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            lr = float(self.arg.base_lr) * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError('Unknown optimizer')

    def save_to_checkpoint(self, state_dict, filename='weights'):
        if not os.path.exists(self.arg.model_saved_dir):
            os.makedirs(self.arg.model_saved_dir)
        link_name = f'{self.arg.model_saved_dir}/{filename}.pt'
        save_name = f'{self.arg.model_saved_dir}/temp_{filename}.pt'
        if os.path.exists(link_name):
            torch.save(state_dict, save_name)
            os.replace(save_name, link_name)
        else:
            torch.save(state_dict, link_name)

    def load_from_checkpoint(self, path=None):
        if path is None:
            path = self.arg.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f'path of checkpoint does not exist: {path}')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if hasattr(self.model, 'module') and isinstance(self.model.module, torch.nn.Module):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.load_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr = self.optimizer.param_groups[0]['lr']
        self.global_step = checkpoint['global_step']
        self.max_acc = checkpoint['max_acc']
        # self.max_test_acc = checkpoint['max_test_acc']
        # self.tester.max_acc = self.max_test_acc
        self.global_epoch = checkpoint['global_epoch']
        print(f'loaded checkpoint from {path}')

    def train(self, epochs, dataloader=None, model=None, is_master=True):
        if model is not None:
            self.model = model
        elif self.model_type == 'MF':
            self.model = MF.Model(graph=graph.Graph())
        elif self.model_type == 'CTR':
            self.model = CTR.Model(graph=graph.Graph())
        elif self.model_type == 'TEG':
            self.model = TEG.Model(graph=graph.Graph())
        else:
            raise ValueError(f'The model_type is not supported: {self.model_type}')
        if dataloader is None:
            self.dataloader = DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                pin_memory=True,
            )
        else:
            self.dataloader = dataloader
        sampler = self.dataloader.sampler if dataloader is not None else None
        if os.path.exists(self.arg.model_path):
            self.load_from_checkpoint()
        else:
            self.model.to(self.device)
            self.load_optimizer()
        self.model.train()
        if is_master:
            self.print_log(f'\t===== training from global steps {self.global_step} =====')
            self.train_writer = SummaryWriter(os.path.join(self.arg.log_dir, 'train'), 'train')
        mean_acc = 0
        for epoch in range(self.global_epoch, epochs):
            loss_value = []
            acc_value = []
            lr = self.adjust_learning_rate(epoch)
            if is_master and lr != self.lr:
                print(f'\tadjusted learning rate from [{self.lr:.6f}] to [{lr:.6f}]')
            if lr != self.lr and os.path.exists(f'{self.arg.model_saved_dir}/best_test_weights.pt'):
                global_epoch = self.global_epoch
                global_step = self.global_step
                max_acc = self.max_acc
                max_test_acc = self.tester.max_acc
                self.load_from_checkpoint(f'{self.arg.model_saved_dir}/best_test_weights.pt')
                if is_master:
                    print(f'\tadjusted learning rate from [{self.lr:.6f}] to [{lr:.6f}]')
                self.global_epoch = global_epoch
                self.global_step = global_step
                self.max_acc = max_acc
                self.tester.max_acc = max_test_acc
                self.adjust_learning_rate(epoch)
            self.lr = self.optimizer.param_groups[0]['lr']
            if sampler is not None:
                sampler.set_epoch(epoch)
            for data, label in tqdm(self.dataloader, desc='Training progress epoch {}'.format(epoch)) \
                    if is_master else self.dataloader:
                if is_master:
                    self.global_step += 1
                # data [N, 12, 300, 17, 2]
                data = torch.as_tensor(data, dtype=torch.float32, device=self.device).detach()
                data = data[:,self.data_idx:self.data_idx+3,:]
                # label [N,]
                label = torch.as_tensor(label, dtype=torch.int64, device=self.device).detach()
                # forward
                output = self.model(data)
                loss = self.loss(output, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_value.append(loss.data.item())
                value, predict_label = torch.max(output.data, 1)
                now_acc = torch.mean((predict_label == label.data).float()).item()
                acc_value.append(now_acc)
                # statistics
                # if is_master:
                #     self.train_writer.add_scalar('acc', now_acc, self.global_step)
                #     self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                #     self.lr = self.optimizer.param_groups[0]['lr']
                #     self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # save the best model
            if self.arg.distributed:
                mean_acc = torch.Tensor([np.mean(acc_value),]).to(self.device)
                torch.distributed.all_reduce(mean_acc, op=torch.distributed.ReduceOp.SUM)
                mean_acc /= torch.distributed.get_world_size()
                mean_loss = torch.Tensor([np.mean(loss_value),]).to(self.device)
                torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
                mean_loss /= torch.distributed.get_world_size()
                mean_acc = mean_acc.item()
                mean_loss = mean_loss.item()
            else:
                mean_acc = np.mean(acc_value)
                mean_loss = np.mean(loss_value)
            if is_master:
                # save model after one epoch
                self.save_to_checkpoint(self.state_dict())

                # logging
                self.train_writer.add_scalar('acc', mean_acc, self.global_epoch)
                self.train_writer.add_scalar('loss', mean_loss, self.global_epoch)
                self.train_writer.add_scalar('lr', self.lr, self.global_epoch)
                if mean_acc > self.max_acc:
                    self.max_acc = mean_acc
                    self.save_to_checkpoint(self.state_dict(), f'weights_acc_{self.max_acc:.4f}')
                    self.save_to_checkpoint(self.state_dict(), f'best_weights')
                self.print_log(f'Training epoch: {epoch}')
                self.print_log(f'\tMean training loss: {mean_loss:.4f}')
                self.print_log(f'\tMean training  acc: {mean_acc:.4f}')
                self.print_log(f'\t Max training  acc: {self.max_acc:.4f}')
                # testing
                self.tester.test(epoch)
                if self.tester.max_acc > self.max_test_acc:
                    self.save_to_checkpoint(self.state_dict(), 'best_test_weights')
                self.max_test_acc = self.tester.max_acc
                self.print_log(f'\t============ global steps {self.global_step} ============')

            self.global_epoch += 1

        if is_master:
            self.save_to_checkpoint(self.state_dict(), f'last_weights_{mean_acc:.4f}')

def train_distributed(replica_id, replica_count, port, arg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=replica_id, world_size=replica_count)
    leaner = Leaner(arg)
    leaner.device = torch.device('cuda', replica_id)
    torch.cuda.set_device(leaner.device)
    dataset = Feeder(**arg.train_feeder_args, is_master=replica_id==0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=2,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    model = Model(graph=graph.Graph(),
                  graph_args=arg.model_args['graph_args']).to(leaner.device)
    model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
    leaner.train(arg.num_epoch, dataloader, model, is_master=replica_id==0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skelton-based Action Recognition')
    parser.add_argument('--config_path',
                        default='./config/params.yaml',
                        help='path of the config file')
    parser.add_argument('--epoch')
    p = parser.parse_args()
    if p.config_path is not None:
        with open(p.config_dir, 'r') as f:
            # default_arg = yaml.load() 会报错
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    seed = random.randint(0, int(1e9))
    setup_seed(seed)
    print_log(arg.log_dir, f'seed is set to: {seed}')
    leaner = Leaner(arg)
    if not arg.distributed:
        arg.device_count = 1
    if arg.device_count > torch.cuda.device_count():
        raise ValueError('Not enough GPUs in your system')
    replica_count = torch.cuda.device_count()
    if arg.distributed and arg.device_count > 1:
        if arg.batch_size % arg.device_count != 0:
            raise ValueError(f'Batch size {arg.batch_size} is not evenly divisble by # GPUs {arg.device_count}.')
        arg.batch_size = arg.batch_size // arg.device_count
        port = _get_free_port()
        spawn(train_distributed, args=(arg.device_count, port, arg), nprocs=arg.device_count, join=True)
    else:
        leaner.train(arg.num_epoch)

