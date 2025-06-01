# 水果图像分类器

该项目使用 PyTorch 构建了一个卷积神经网络(CNN)，用于对水果图像进行分类。模型在 [Fruits-360 数据集](https://www.kaggle.com/datasets/moltean/fruits) 上训练，能够识别 131 种不同种类的水果。

## 功能特点

- 使用自定义 CNN 架构进行图像分类
- 支持 GPU 加速训练
- 实现学习率调度和梯度裁剪
- 包含数据预处理和可视化
- 提供 Gradio 网页界面进行交互式预测
- 使用Adam优化器替换默认的梯度下降法，Adam优化器具有自动调整学习率等方面特点

## 数据集

Fruits-360 数据集包含 131 种水果的 90,483 张图像：

- 训练集：67,692 张图像
- 测试集：22,688 张图像

数据集目录结构：

```
fruits-360/
├── Training/
│   ├── Apple Braeburn/
│   ├── Apple Golden 1/
│   └── ... (131 个子目录)
└── Test/
    ├── Apple Braeburn/
    ├── Apple Golden 1/
    └── ... (131 个子目录)
```

## 依赖项

- PyTorch
- torchvision
- matplotlib
- tqdm
- gradio
- PIL

安装依赖：

```bash
pip install torch torchvision matplotlib tqdm gradio
```

## 使用说明

1. 从 Kaggle 下载数据集：https://www.kaggle.com/datasets/moltean/fruits
2. 解压数据集到项目根目录
3. 运行完整代码：

```python
python fruit_classifier.py
```

4. 训练完成后，Gradio 界面将自动启动
5. 在浏览器中打开显示的 URL 进行交互式测试

## 模型架构

```python
Fruit360CnnModel(
  (network): Sequential(
    (0): Conv2d(3, 16, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))                   # 卷积层（输入3，输出16，卷积核大小2*2，步长1，边界1）
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    # 归一化
    (2): ReLU()                                                                             # 激活函数ReLU
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)         # 池化层：最大池化
    (4): Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))                  # 以下到11都是以卷积 -> 归一化 -> 激活函数 -> 池化次序排列
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
    (12): Flatten(start_dim=1, end_dim=-1)                                                  # 展平为向量
    (13): Dropout(p=0.3, inplace=False)                                                     # 丢弃30%的神经元防止过拟合
    (14): ReLU()                                                                            # 激活函数ReLU
    (15): Linear(in_features=1600, out_features=131, bias=True)                             # 全连接层64*5*5，输出131个类别
  )
)
```

## 训练配置

- 优化器: Adam
- 学习率: 0.01 (使用 one-cycle 策略)
- 批量大小: 256
- 训练轮数: 4
- 梯度裁剪: 0.1
- 权重衰减: 1e-4

## 结果

训练完成后，模型在验证集上达到约 95% 的准确率。

## 界面预览

![Gradio 界面](preview.gif)

## 作者

[CongYu404]

## 项目结构说明

```
fruit-classifier/
├── fruits-360/             # 数据集（需自行下载）
│   ├── Training/           # 训练集
│   └── Test/               # 测试集
├── cnn.pth                 # 训练好的模型
├── Temp/                   # 临时目录（Gradio预测用）
│   └── test/
│       └── test.jpg        # 临时图像
└── fruit_classifier.py     # 主代码文件
└── fruit_classifier.ipynb  # notebook代码文件
```

## 使用提示

1. 在运行前确保已下载数据集并放在正确位置
2. 首次运行会训练模型并保存为 `cnn.pth`
3. 后续运行可直接加载模型进行预测
4. Gradio 界面默认在 `http://localhost:7860` 启动
5. 可通过调整超参数（如学习率、批次大小等）优化模型性能
