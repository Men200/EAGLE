import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/xwwen/EAGLE_test/Llama-2-7b-chat-hf') # 预训练模型的路径
parser.add_argument('--configpath', type=str, default="/home/xwwen/EAGLE_test/EAGLE/eagle/train/llama_2_chat_7B_config.json") # 模型配置文件的路径
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='/home/xwwen/EAGLE_test/training_data/sharegpt_0_67_mufp16/') # 训练数据路径
parser.add_argument('--cpdir', type=str, default='/home/xwwen/EAGLE_test/checkpoints') # checkpoint存储路径，以便在训练中断时可以恢复
args = parser.parse_args()

"""通过预训练模型的隐藏状态来训练一个自回归头"""

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    # "num_epochs": 20,
    "num_epochs": 512,    
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000, # 预热步数
    "total_steps": 800000,
    "cross_entropy_w": 0.1,
    "smooth_L1_w": 1.0,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    # "data_noise": True, # 是否对数据添加噪声
    "data_noise": False,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that 
    # the larger the setting, the more training data is used, 
    # and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}
import json
from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
"""
Hugging Face的Accelerate库的一个用于简化分布式训练和混合精度训练的工具库，特别适合在多个GPU或TPU上进行训练。
在下列设置中启用混合精度训练，使用BF16格式，并设置梯度累计步数。后续还有很多接口，诸如accelerator.prepare、
save_state、load_state、clip_grad_value_
"""
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

from ..model.cnets import Model
from ..model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

def init_head():
    """用预训练模型的语言头参数初始化一个head，并冻结，要靠它将目标中间特征转换为目标token"""

    # 加载预训练模型的配置
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    # 初始化一个线性层，用于将隐藏状态映射到词汇表大小
    head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        # 安全地打开.safetensors文件
        with safe_open(os.path.join(args.basepath, head_path),
                    framework="pt", # 指定PyTorch作为框架
                    device="cpu") as f: # 指定在CPU上加载
            
            # 获取lm_head.weight的切片张量，避免一次性加载整个文件，提高效率
            tensor_slice = f.get_slice("lm_head.weight")

            # 获取lm_head.weight的维度，词汇表大小×隐藏层维度（这个张量切片操作显得有些多余啊）
            # vocab_size, hidden_dim = tensor_slice.get_shape()
            # tensor = tensor_slice[:, :hidden_dim].float()
            tensor = tensor_slice.float()
    except:
        # 如果.safetensors加载失败，改用.bin
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]

        # 使用torch.load加载.bin格式的权重，挑出语言建模头部分转为浮点数张量
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head.weight.data = tensor
    # 进入评估模式，使用固定的均值/方差，关闭dropout，不影响反向传播
    head.eval()
    # 冻结head的所有参数，防止梯度更新
    for param in head.parameters():
        param.requires_grad = False
    return head

def list_files(path):
    """遍历指定目录下的所有文件，并返回包含文件路径的列表"""

    datapath = []
    # os.walk(path)自动递归遍历文件系统，不断返回
    # 当前遍历的目录路径、当前目录下的所有子目录名称列表、文件名称列表
    for root, directories, files in os.walk(path):
        for file in files:
            # 构造完整的文件路径
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class AddGaussianNoise:
    """给“hidden_state_big”张量添加高斯噪声"""

    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # torch.randn(tensor.size())生成与tensor形状相同的随机数张量，值服从N(0,1)
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class AddUniformNoise:
    """给“hidden_state_big”张量添加均匀噪声"""

    def __init__(self, std=0.0):
        self.std = std # 标准差用于控制噪声大小

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # torch.rand_like(tensor) - 0.5 生成 (-0.5, 0.5) 之间的均匀噪声
        # self.std控制噪声强度
        # 512 / tensor.shape[1]归一化，确保不同长度的hidden_state_big有相似的噪声比例
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class CustomDataset(Dataset):
    """自定义数据集类，用于加载和处理数据"""

    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform # 预处理转换函数（如噪声）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = torch.load(self.data[index])
        new_data = {}
        # 将读取的input_ids、hidden_state、loss_mask都裁剪到max_len
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        length = hidden_state.shape[1]

        # 注意力掩码全1
        attention_mask = [1] * length

        # 损失掩码最后一个位置设0
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        # 提前一个时间步的token序列，用0填充
        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        # 用于计算hidden_state的目标值
        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)

        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["target"] = target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:
    """自定义的批处理方式，即如何将多个数据样本组合成一个批次"""

    # 对3D张量(batch × n × S)进行填充，使第二个维度n扩展到N（序列长度）
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    # 用于对2D张量进行填充
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    # 该类的主要入口，负责处理一个batch的数据，将所有样本对齐到相同长度
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 计算最大序列长度
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        # 所有input_ids拼成(B, max_length)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        # 所有batch_hidden_states拼成(B, max_length, hidden_dim)
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        # 所有target拼成(B, max_length, hidden_dim)
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        
        # 两掩码均一维列表，用零填充，最终形状(B, max_length)
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)

        # 将所有处理好的数据封装到一个字典中，方便后续用于DataLoader进行批量计算
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

def top_kaccuracy(output, target, topk=(1,)):
    """衡量模型预测的top-k个最可能的结果中是否包含真实标签"""

    # output.shape (bs, num_classes), target.shape (bs, )
    # 模型的输出张量，每个样本对不同类别的预测分数；真实标签，每个样本的正确类别索引
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 取每个样本预测分数前maxk个最大值的索引，返回值忽略分数，保留索引，形为(batch_size, maxk)
        _, pred = output.topk(maxk, 1, True, True)
        # pred转置成(maxk, batch_size)
        pred = pred.t()
        # 将target扩展成和pred同形状后，与pred进行比对，返回True/False矩阵
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 取前k行，将其拉成一维向量，转换为浮点数1.0/0.0，统计预测正确的样本数
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask, head, criterion):
    """计算交叉熵损失、平滑L1损失"""

    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head) # 交计算log-softmax
    plogp = target_p * out_logp
    cross_entropy_loss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5) # 防止除零错误

    smooth_L1_loss = criterion(predict, target)
    smooth_L1_loss = torch.sum(torch.mean(loss_mask * smooth_L1_loss, 2)) / (loss_mask.sum() + 1e-5)

    return cross_entropy_loss, smooth_L1_loss, out_head

"""下面两个函数只在评估时使用"""
@torch.no_grad()
def generate(model, hidden_states, input_ids, head, max_length=4, use_cache=True):
    """生成长为max_length的预测序列"""
    if use_cache:
        # past_key_values用于缓存前面已计算的Transformer注意力机制的键值对
        past_key_values = None
        for i in range(max_length):
            if past_key_values != None:
                # 后续步骤则使用上一次的隐藏状态和生成的token进行推理
                out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                                    use_cache=True)
            else:
                # 若past_key_values为空，即第一步，则使用hidden_states和input_ids进行计算
                out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
            
            last_hidden = out_hidden[:, -1:] # 取最后一个时间步的隐藏状态
            last_headout = head(last_hidden) # 通过输出头转换为logits
            token = torch.argmax(last_headout, dim=-1) # 取最大概率的token
            input_ids = torch.cat((input_ids, token), dim=1) # 拼到已有的token序列

    else:
        # 不用缓存的方法暂未实现
        raise NotImplementedError

    return input_ids

@torch.no_grad()
def get_steps_acc(model, data, head, max_length=5):
    """计算模型逐步生成预测序列的逐步准确率"""

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    # 需要预测的token总数
    total = [0 for _ in range(max_length)]
    # 正确预测数
    correct = [0 for _ in range(max_length)]

    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0: # 这个位置没有损失计算，则无需预测直接跳过
            continue

        # 前pre_len个隐藏状态和token
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]

        # 预测pre_len:之后max_length个token
        outs = generate(model, pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]

        # 遍历批量中所有样本
        for bid in range(bs):
            # 计算不同长度的预测准确度
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0: # 没有损失则不参与计算
                    break
                if pre_len + k >= seq_len: # 超出序列长度则跳出
                    break
                total[k] += 1 # k处的预测任务数+1
                # 预测正确
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                # 预测错误，后续的预测都算失败
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc

def train_one_epoch(train_loader, model, head, criterion, optimizer, scheduler, is_warmup):
    total_top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train() # 启动目标模型的训练模式

    # 遍历当前批次的所有样本
    for batch_idx, data in enumerate(tqdm(train_loader)): # tqdm进度条显示训练进度

        """使用梯度累加，在小批次下模拟大批量训练
        显存不足，无法一次性处理大批量训练时，采用梯度累计，即
        不是每个批次都更新参数，而是累计多个批量的梯度后更新一次，
        以模拟大批量训练，提高稳定性，同时避免显存溢出
        """
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            # 前向传播
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])

            # 计算目标值（无需梯度跟踪）
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            # 计算损失（token分布的交叉熵损失+隐藏状态的平滑L1损失的加权）（要想反向传播就不能不跟踪loss的梯度）
            loss_mask = data["loss_mask"][:, :, None]
            cross_entropy_loss, smooth_L1_loss, out_head = compute_loss(data["target"], target_p, predict, loss_mask, head, criterion)
            loss = train_config["cross_entropy_w"] * cross_entropy_loss + train_config["smooth_L1_w"] * smooth_L1_loss

            # 反向传播
            accelerator.backward(loss) # 计算损失梯度
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"]) # 梯度裁剪防爆炸
            optimizer.step() # 优化器根据当前梯度和优化器配置更新模型参数
            if is_warmup:
                scheduler.step() # 更新学习率调度器

        # 计算准确率（无需梯度跟踪）
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2) # 沿着第二维（类别维度）找到最大值
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item() # 总样本数
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item() # 正确预测的数量
            total += ct
            correct += cc

            # 累积所有批次的top-1、top-2、top-3准确率
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            top_kacc = top_kaccuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(top_kacc)):
                total_top_3acc[top_i] += top_kacc[top_i]

        # 记录日志（wandb监控）
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"], 
                "train/cross_entropy_loss": cross_entropy_loss.item(),
                "train/smooth_L1_loss": smooth_L1_loss.item(), 
                "train/loss": loss.item(), 
                "train/acc": cc / ct
            }
            # 在W&B中，train/acc和train/top_1_acc的图是一样的
            for id, i in enumerate(top_kacc):
                logdict[f'train/top_{id + 1}_acc'] = top_kacc[id].item() / ct

            wandb.log(logdict)

        del cross_entropy_loss, smooth_L1_loss
        epoch_loss += loss.item()
        num_batches += 1

    # gather_for_metrics分布式评估，在多个设备上并行计算评估指标
    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda() # 将标量转换成张量，并移动到GPU上
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    total_top_3acc = accelerator.gather_for_metrics(total_top_3acc)
    epoch_loss /= num_batches

    return correct, total, total_top_3acc, epoch_loss

def eval_this_epoch(test_loader, model, head, criterion):
    total_top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.eval() # 将模型设为评估模式，关闭dropout和batch normalization的随机性

    steps_5acc = [[] for i in range(5)]
    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad(): # 禁用梯度，无需存储反向传播所需中间结果，减少内存占用并加速计算

            # 计算前10个批次的逐步预测准确率
            if batch_idx < 10:
                acces = get_steps_acc(model, data, head, max_length=5)
                for i in range(len(acces)):
                    steps_5acc[i].append(acces[i])

            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            target_head = head(data["target"])
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

            loss_mask = data["loss_mask"][:, :, None]
            cross_entropy_loss, smooth_L1_loss, out_head = compute_loss(data["target"], target_p, predict, loss_mask, head, criterion)
            loss = train_config["cross_entropy_w"] * cross_entropy_loss + train_config["smooth_L1_w"] * smooth_L1_loss

            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            total += ct
            correct += cc

            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            top_kacc = top_kaccuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(top_kacc)):
                total_top_3acc[top_i] += top_kacc[top_i]

        epoch_loss += loss.item()
        num_batches += 1

    mean_acces = []
    # steps_5acc形状为5（步）×10（epoch）
    for id, i in enumerate(steps_5acc):
        mean_acc = np.array(i).mean()
        mean_acc = torch.tensor(mean_acc).cuda()
        mean_acces.append(mean_acc)

    mean_acces = accelerator.gather_for_metrics(mean_acces)
    if accelerator.is_local_main_process:
        for id, i in enumerate(mean_acces):
            mean_acc = i.mean().item()
            wandb.log({f"test/{id}_acc": mean_acc})

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    total_top_3acc = accelerator.gather_for_metrics(total_top_3acc)
    epoch_loss /= num_batches

    return correct, total, total_top_3acc, epoch_loss

def main():
    # 是否要在数据中加入噪声以增强多样性
    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    # 从datapath目录中获取所有数据的文件路径，95%作训练集，5%作测试集
    datapath = list_files(train_config["datapath"])
    np.random.shuffle(datapath) # 打乱数据顺序
    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]

    # 构建数据集（训练时可选地加入噪声）
    traindataset = CustomDataset(traindatapath, transform=aug)
    testdataset = CustomDataset(testdatapath)
    # 包装成可迭代的批次数据，num_workers为并行加载数据的线程数
    """
    在PyTorch中，pin_memory=True是一个用于优化数据加载性能的设置。默认情况下为False：
    数据一般存储在CPU的可分页内存（pageable memory）中，但GPU只能高效地读取固定的锁页内存（pinned memory）。
    故系统会先将数据从pageable memory复制到pinned memory，再拷贝到GPU，多余步骤降低了数据传输速度，
    开启pin_memory后PyTorch会直接在pinned memory中存储数据。
    在测试阶段数据加载通常是顺序的，且每个batch只需加载一次，无反向传播和参数更新，数据加载速度对整体性能的
    影响较大；在训练阶段数据加载随机，且每个batch需多次加载，训练过程包括前向传播、反向传播、参数更新，数据
    加载的时间通常被计算时间覆盖，而且不将训练数据加载到锁页内存中也可以减少主机内存的占用。
    """
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

    # 只在主进程上创建args.cpdir目录，用于保存模型权重
    if accelerator.is_main_process:
        if not os.path.exists(args.cpdir):
            os.makedirs(args.cpdir)

    # 加载模型配置
    config = EConfig.from_pretrained(train_config["config_path"])
    # 初始化Model，加载预训练嵌入
    model = Model(config, load_emb=True, path=args.basepath)

    # 平滑L1损失（又称Huber损失）（介于绝对误差损失和均方误差之间的损失函数）
    criterion = nn.SmoothL1Loss(reduction="none")
    # AdamW优化器，控制梯度一阶和二阶矩估计
    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

    # 学习率调度器
    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    head = init_head()

    # 前num_warmup_steps线性增加学习率，然后线性下降，以避免训练初期学习率过高导致的不稳定
    if is_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)

        # 自动检测可用设备，将模型、优化器、数据，甚至调度器移到正确的设备上，自动处理DP、PP等分布式训练策略。
        model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader, scheduler # 若使用Warmup，还要包含scheduler
        )
    else:
        model, head, optimizer, train_loader, test_loader = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader
        )

    # 加载之前保存的状态，从中断点继续训练
    # accelerator.load_state("checkpoints/state_5")

    # 训练循环
    for epoch in range(num_epochs + 1):

        correct, total, total_top_3acc, epoch_loss = train_one_epoch(train_loader, model, head, criterion, optimizer, scheduler, is_warmup)

        # 在主进程中记录当前批次的训练日志
        if accelerator.is_local_main_process:
            # 不同k的top-k准确度
            for id, i in enumerate(total_top_3acc):
                wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
            # 当前epoch的损失值和准确率
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
        

        # 每隔5个epoch测试一次
        if (epoch + 1) % train_config["save_freq"] == 0:
            correct, total, total_top_3acc, epoch_loss = eval_this_epoch(test_loader, model, head, criterion)

            if accelerator.is_local_main_process:
                for id, i in enumerate(total_top_3acc):
                    wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
                print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
                print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
                wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})

                """
                保存当前训练状态到指定目录，包括优化器状态optimizer.bin、模型主权重pytorch_model.bin、
                模型额外权重pytorch_model_1.bin、随机数生成器状态random_state_0.pkl、学习率调度器状态
                scheduler.bin……便于后续恢复训练
                """
                accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
            
if __name__ == "__main__":
    if accelerator.is_main_process:
        import wandb
        """使用W&B.init()来启动一个实验，它会自动追踪超参数、训练过程、结果等"""
        wandb.init(project="ess", entity="3201806824-tsinghua-university", config=train_config)

    main()

    if accelerator.is_main_process:
        wandb.finish()