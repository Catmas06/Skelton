import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.module_ta import Multi_Head_Temporal_Attention
from model.module_cau import unit_tcn_causal


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn_3d(nn.Module):

    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True, win = 3):

        super(unit_gcn_3d,self).__init__()

        self.win = win
        self.num = A.shape[0]

        def build_spatial_temporal_graph(A_binary, window_size):
            assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
            V = len(A_binary)
            V_large = V * window_size
            A_large = np.tile(A_binary, (window_size, window_size)).copy()
            return A_large

        A_tile = [build_spatial_temporal_graph(A[i],self.win) for i in range(self.num)]
        self.A_tile = np.stack(A_tile, axis=0)
        self.gcn = unit_gcn_qkv(in_channels,out_channels,self.A_tile,adaptive=adaptive,attention=attention)

    def forward(self,x):

        B,C,T,V = x.shape
        x = x.reshape(B,C,-1,V*self.win)
        x = self.gcn(x)
        x = x.reshape(B,-1,T,V)

        return x


class unit_gcn_qkv(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(unit_gcn_qkv, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.ffn = nn.ModuleList()
        for i in range(self.num_subset):
            self.ffn.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.emb_q = nn.ModuleList()
            self.emb_k = nn.ModuleList()
            for i in range(self.num_subset):
                self.emb_q.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.emb_k.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

            # self.bn = nn.BatchNorm2d(out_channels)
            # bn_init(self.bn, 1)
        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.ffn[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.num_subset):
                q =  self.emb_q[i](x)
                q = q.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                k =  self.emb_k[i](x)
                k = k.view(N, self.inter_c * T, V)
                att = self.tan(torch.matmul(q, k) / q.size(-1))  # N V V
                att = A[i] + att * self.alpha
                #v = x.view(N, C * T, V)
                v = x.reshape(N, C * T, V)
                v = torch.matmul(v, att).view(N, C, T, V)
                z = self.ffn[i](v)
                y = z + y if y is not None else z
        else:
            #A = self.A.cuda(x.get_device()) * self.mask
            A = self.A.cuda(x.get_device())
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.ffn[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            # unified attention
            # y = y * self.Attention + y
            # y = y + y * ((a2 + a3) / 2)
            # y = self.bn(y)
        return y


class TCNC_GCN3D_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCNC_GCN3D_unit, self).__init__()
        self.gcn1 = unit_gcn_3d(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        #self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = unit_tcn_causal(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):

        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

        return y


class TCNC_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCNC_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn_qkv(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        #self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = unit_tcn_causal(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):

        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

        return y


class TA_GCN3D_unit(nn.Module):
    def __init__(self, in_channels, out_channels, T, A, stride=1, residual=True, adaptive=True, attention=True,inherent=1,head=4,norm='ln',dropout=0.0,with_cls_token=0,pe=1):
        super(TA_GCN3D_unit, self).__init__()
        self.gcn1 = unit_gcn_3d(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        #self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.ta1 = Multi_Head_Temporal_Attention(out_channels,head,T,A,inherent,norm,dropout,with_cls_token,pe)
        #self.tcn1 = unit_tcn_causal(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):

        z,cls_token = self.ta1(self.gcn1(x))

        z += self.residual(x)
        y = self.relu(z)

        #return y,cls_token
        return y

class Model(nn.Module):
    def __init__(self, num_class=155, num_point=17, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, attention=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        # else:
            # Graph = import_class(graph)
            # self.graph = Graph(**graph_args)
        self.graph = graph

        A = self.graph.A
        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        #3D-TA all
        if 1:
            self.l1 = TCNC_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive, attention=attention)
            self.l2 = TCNC_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
            self.l3 = TCNC_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
            self.l4 = TCNC_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
            self.l5 = TCNC_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
            self.l6 = TA_GCN3D_unit(128, 128, 30, A, adaptive=adaptive, attention=attention,inherent=0,head=1,dropout=0,pe=0)
            self.l7 = TA_GCN3D_unit(128, 128, 30, A, adaptive=adaptive, attention=attention,inherent=0,head=1,dropout=0,pe=0)
            self.l8 = TCNC_GCN3D_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
            self.l9 = TA_GCN3D_unit(256, 256, 15, A, adaptive=adaptive, attention=attention,inherent=0,head=1,dropout=0.0,pe=0)
            self.l10 = TA_GCN3D_unit(256, 256, 15, A, adaptive=adaptive, attention=attention,inherent=0,head=1,dropout=0.0,pe=0)
            output_emb = 256

        self.emb_dim = num_point * 256

        self.fc = nn.Linear(output_emb, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        # x = x.view(N, C * T * V * M)
        # x = self.data_bn(x)
        x = x.view(N, C, T, V, M).permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        #N,C,T,V
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)

        return x
        #if cls_token is not None:
            #cls_token = rearrange(cls_token, '(n m) t d -> n m (t d) ',m=2)
            #cls_token = reduce(cls_token, 'n m d -> n d','mean')
            #z = self.mlp_head(cls_token)
            #return z
        #else:
            #return x


if __name__ == '__main__':
    pass