import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr
from PIL import Image

# os.environ解决部分系统上的库冲突的问题，例如macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def to_device(
        data:torch.Tensor,
        device:torch.device
    ) -> torch.Tensor:
    """ 
    将数据移动到GPU上进行训练，对所有数据递归处理，返回列表，再进行数据迁移至GPU/CPU
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data] 
    return data.to(device, non_blocking=True)
 
def accuracy(outputs, labels):
    """计算准确率"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()                                                            # 关闭梯度计算
def evaluate(model, val_loader):
    """评估模型在验证集上的性能"""
    model.eval() # 将模型切换为评估模式
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
 
def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
 
def fit_one_cycle(
        epochs,                  # 周期
        max_lr,                  # 学习率
        model,                   # 模型
        train_loader,            # 训练集
        val_loader,              # 测试集
        weight_decay=0,          # 默认权重为0
        grad_clip=None,          # 默认梯度衰减为0
        opt_func=torch.optim.SGD # 默认调用随机梯度，以防调用Adam错误
    ):
    """单周期学习率调度函数"""

    torch.cuda.empty_cache()                                                # 清理GPU缓存
    history = []                                                            # 保存历史
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    # 设置单周期学习率调度器
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr, 
        epochs=epochs,
        steps_per_epoch=len(train_loader)
        )
    
    # 训练阶段
    for epoch in range(epochs):
        model.train()       # 将模型设置为训练模式
        train_losses = []   # 记录损失
        lrs = []            # 记录学习率变化

        # 可视化进度条进行批次训练
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()     # 反向传播
            
            # 梯度裁剪，防止梯度爆炸
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()        # 更新权重
            optimizer.zero_grad()   # 清除梯度
            
            lrs.append(get_lr(optimizer))   # 记录学习率
            sched.step()                    # 更新学习率
        
        # 对模型进行验证
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
 

def predict_image(
        img:torch.Tensor,
        model:torch.nn.Module
    ) -> str: 
    """
    传入图像张量，以及要使用的模型，模型将会把数据迁移至GPU上进行预测，
    并返回字符类型131类水果中的一类
    """
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return train.classes[preds[0].item()]

def predict(img):
    model = torch.load("cnn.pth")
    model.eval()
    if isinstance(img, str):        # 判断传入的img是否为字符类型
        img_proc = Image.open(img)
    else:
        return "FileRequestError"
    img_proc = img_proc.resize((100, 100))  # 对图片进行强制缩放处理
    img_proc.save("./Temp/test/test.jpg")   # 保存到本地以便导入到模型预测
    img_md = ImageFolder("./Temp", transform=ToTensor())
    i, l = img_md[0]
    result = predict_image(i, model) # 调用预测函数
    return result

class DeviceDataLoader():
    """自动化将数据集移动到GPU/CPU上"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """迭代器，迭代时生成已移动到设备的数据批次"""
        for b in self.dl: 
            yield to_device(b, self.device)
 
    def __len__(self):
        """返回批次数量"""
        return len(self.dl)

class ImageBase(nn.Module):
    """图像分类模型基类"""
    def training_step(self, batch):
        """训练步骤"""
        images, labels = batch 
        out = self(images)                      # 生成预测
        loss = F.cross_entropy(out, labels)     # 计算损失
        return loss
    
    def validation_step(self, batch):
        """验证步骤"""
        images, labels = batch 
        out = self(images)                      # 生成预测
        loss = F.cross_entropy(out, labels)     # 计算损失
        acc = accuracy(out, labels)             # 计算损失率
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        """验证周期结束处理"""
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # 合并损失率
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # 合并准确率
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        """每周期训练信息进行打印输出"""
        print(f"Epoch [{epoch}], train_loss: {result["train_loss"]:.4f}, val_loss: {result["val_loss"]:.4f}, val_acc: {result["val_acc"]:.4f}")

class CnnModel(ImageBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential( # 神经网络框架？容器
            nn.Conv2d(3, 16, kernel_size=2, padding=1), 
            nn.BatchNorm2d(16), # 对16个输出进行归一化
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 池化100x100 -> 50x50 最终输出为：16x50x50
 
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), 
            nn.BatchNorm2d(32), # 对32个输出进行归一化
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 池化50x50 -> 25x25 最终输出为：32x25x25
 
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64), # 对64个输出进行归一化
            nn.ReLU(),
            nn.MaxPool2d(5, 5), # 池化25x25 -> 5x5 最终输出为：64x5x5
 
            nn.Flatten(),       # 展平为向量
            nn.Dropout(0.3),    # 丢弃30%的神经元防止过拟合
            nn.ReLU(),
            nn.Linear(64*5*5, 131) # 全连接层
        )
        
    def forward(self, xb):
      """向前传播"""
      return self.network(xb)

demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type="filepath"),
    title="水果识别分类器",
    outputs = gr.Text(type="text"),
    description="上传水果图像，模型将识别其种类（支持131种水果）"
)

if __name__ == '__main__':
     
    model = CnnModel() # 创建模型实例

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train = ImageFolder("./fruits-360/Training", transform=ToTensor())
    test = ImageFolder("./fruits-360/Test", transform=ToTensor())

    torch.manual_seed(20) 
    val_size = round(len(train) * 0.2)
    train_size = round(len(train) - val_size)
    train_ds, val_ds = random_split(train, [train_size, val_size])

    batch_size = 256
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True) 
    val_dl = DataLoader(val_ds, batch_size * 2, pin_memory=True)

    model = to_device(model, device)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(val_dl, device)

    epochs = 4                  # 训练次数（周期）
    max_lr = 0.01               # 最大学习率
    grad_clip = 0.1             # 梯度裁剪（防止梯度爆炸）
    weight_decay = 0.001        # 权重衰减
    opt_func = torch.optim.Adam # 启用Adam优化器
    history = []

    history += fit_one_cycle(
        epochs, 
        max_lr, 
        model,
        train_dl, 
        valid_dl, 
        grad_clip=grad_clip, 
        weight_decay=weight_decay, 
        opt_func=opt_func
        )
    torch.save(model, "cnnpy.pth")
    demo.launch()