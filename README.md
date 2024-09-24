# skelton

#### 介绍
省赛

#### 软件架构
```
└─config
    ├── params.yaml
        用于存放训练、测试所用参数。所有参数更改都在这里完成
└─data
    ├── train_joint.npy
    ├── train_label.npy
    ├── test_joint_A.npy
    ├── train_label_A.npy
        将省赛/data目录下所有文件解压到此
└─model
    ├── dmodel.py       主要使用的TEGCN模型
    ├── module_cau.py   TEGCN的其他依赖文件
    ├── module_ta.py    TEGCN的其他依赖文件、
└─pre_data  用于对源数据进行预处理的代码目录
    ├── feeder.py       实现了dataset子类，用于存放训练所需数据
    ├── gen_modal.py    用于生成不同模态数据
    ├── graph.py        图的结构定义类
└─utils
    ├── tools.py     各种数据读取方式的函数
    ├── visualize.py    对图结构可视化的函数
└─log*      存放训练中数据
└─output*   存放模型权重文件等训练结果
└─train.py
└─test.py
```

#### 使用说明

1.  将数据解压到data文件夹下，结构如上所示
2.  可以直接运行train.py文件，无需指定参数
3.  按需要在params.yaml中更改参数

#### debug模式
由于数据集过大，模型可设置debug模式。debug模式中仅加载前100项数据，其他结构不变。

config / params.yaml / train_feeder_args / debug: True

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_dev 分支
3.  提交代码
4.  新建 Pull Request
