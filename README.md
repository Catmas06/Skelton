# skelton

#### 介绍
省赛

#### 文件架构
```
└─config
    ├── params.yaml
    ...
        用于存放训练、测试所用参数。所有参数更改都在这里完成
└─data
    └─train     
      train下存放A数据集
        ├── train_joint.npy
        ├── train_label.npy
        ├── test_jointnpy
        ├── train_label.npy
    └─test
        └── test_joint.npy
        
└─model
    ...
└─pre_data  用于对源数据进行预处理的代码目录
    ├── feeder.py       实现了dataset子类，用于存放训练所需数据
    ├── gen_modal.py    用于生成不同模态数据
    ├── graph.py        图的结构定义类
└─utils
    ├── tools.py     各种数据读取方式的函数
    ├── visualize.py    对图结构可视化的函数
└─log*      存放训练中数据
└─output*   存放模型权重文件等训练结果
...
```

#### 使用说明

1.  将数据解压到data文件夹下，结构如上所示
2.  运行`train.py`文件，根据所需模型及模态指定参数
    ```angular2html
    train.py --config_path=./config/mf_j.yaml
    ```
3.  分别完成12次训练后，运行`generate_score.py`。此文件生成 所有的分数文件，
    每个模型生成三个分数文件，分别是A_train、A_test、B_test对应的分数
4. 所有分数文件生成后，运行`mixer.py`。此文件根据label寻找最优超参数，即融合模型参数
5. 将所找到的最优超参赋值给`generate_B_score.py`中的`rate`变量（代码中已给出），
   运行`generate_b_score.py`文件，得到最终分数文件。其位置为：./output/score/final_score.npy

#### 两种方法
1. 一种我们取mf_j,mf_b,ctr_j,ctr_b和teg_j模态，仅融合这5个正确率最高的模态
2. 分别将mf_jm,mf_bm,ctr_jm,ctr_bm训练两次，得到两个模型和分数文件。之后分别计算每个模型在
    A训练集上155个类别的正确率，并将该正确率作为权重乘以各自的分数文件，将两个相同模型的分数文件
    相加并除以2作为最终融合时使用的分数文件。此举旨在增强低正确率模型中的有用部分，防止融合时扰动过大
    ```
   方法二的代码在mix2.py中 方法一的代码在generate_B_score.py中
    ```


#### debug模式
由于数据集过大，模型可设置debug模式。debug模式中仅加载前100项数据，其他结构不变。

config / params.yaml / train_feeder_args / debug: True

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_dev 分支
3.  提交代码
4.  新建 Pull Request
