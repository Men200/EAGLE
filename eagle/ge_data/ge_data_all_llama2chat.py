import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100) # 定义数据集的选取范围（0-100）
parser.add_argument('--index', type=int, default=1) # 指定输出目录的索引
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='/home/xwwen/EAGLE_test/training_data')
args = parser.parse_args()
import os

# 当输入指令设置参数“--gpu_index 0 1 2”时，被传入的args.gpu_index=[0,1,2]，特意去掉前后两括号，
# 便相当于设置“export CUDA_VISIBLE_DEVICES=0,1,2”，限定程序运行时使用的GPU，避免不必要的资源占用
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
# 显示进度条
from tqdm import tqdm
# 自动加载因果语言模型（LLaMA-2）、分词器，支持量化模型加载（如4-bit量化）
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
# 用于加载Hugging Face数据集
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

# llama2-chat-7B权重路径
bigname = "/home/xwwen/EAGLE_test/Llama-2-7b-chat-hf/"
# bigname="/home/hongyanz/scratch/weights/llama2chat/13B"
# bigname = "/home/lyh/weights/hf/llama/7B/"
# smallname = "/home/lyh/weights/hf/llama/7B/"

# 计算两个列表之间的最长公共前缀
def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length

# 数据集加载和预处理
def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    # 加载JSON格式的ShareGPT数据集
    # ds = load_dataset('json', data_files="/home/hongyanz/scratch/data/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    # ds = load_dataset("shareAI/ShareGPT-Chinese-English-90k", cache_dir="/home/xwwen/EAGLE_test/ShareGPT_dataset")
    ds = load_dataset('json', data_files="/home/xwwen/EAGLE_test/ShareGPT_dataset/computer_en_26k.jsonl")
    ds = ds['train']
    ds = ds.shuffle(seed=42) # 确保随机性一致
    ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 4 # 指定4个进程进行数据预处理

    def preprocess_function(examples):
        new_examples = {
            "conversation":[], # 存储对话文本
            "input_ids": [], # token ID
            "loss_mask": [] # 标记哪些token需要计算loss
        }

        # 遍历数据集中的所有对话
        for i in range(len(examples['id'])):

            # 获取LLaMA-2对话模板
            conv = get_conversation_template("llama-2-chat")

            # 设定AI助手的形为准则
            sys_p="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
            conv.system_message=sys_p

            # 确保第一句话是人类发起的对话
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            # 交替处理human和gpt的对话内容
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                # 若当前句子是gpt的回复，要在sentence["value"]前加一个空格
                if sentence["from"]=="gpt":
                    sentence["value"]=" "+sentence["value"]
                conv.append_message(role, sentence["value"])

            # 获取最终格式化后的对话文本
            conversation=conv.get_prompt()

            # if i==56:
            #     print(i)
            # if i==57:
            #     print(i)

            # 若tokenizer没有pad_token_id，则将其设置为unk_token_id（未知token ID）
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            # 将对话转成token ID
            input_ids = tokenizer(
                conversation,
                return_tensors="pt", # 结果返回PyTorch张量
                max_length=2048,
                truncation=True,
            ).input_ids[0]

            # 创建一个形同input_ids的张量loss_mask，初始值全为1，默认所有token都会计算loss
            loss_mask=torch.ones_like(input_ids)
            #print(i)

            # sep是AI回复前的分隔符，例如“GPT: ”
            sep = conv.sep + conv.roles[1] + " "

            # 实际非填充的token个数
            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            # 将对话拆分为不同轮次
            turns = conversation.split(conv.sep2)

            # 忽略系统消息部分的loss计算
            cur_len = 1
            loss_mask[:cur_len] = 0

            # 处理每轮对话
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids) # 当前对话的token长度

                parts = turn.split(sep) # 拆分成用户输入和AI回复
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # 用户输入部分的token长度

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     instruction_len -= 1

                # Ignore the user instructions（忽略用户输入部分的loss计算）
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len+=2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            # 忽略padding位置的loss计算
            loss_mask[cur_len:] = 0

            # 把格式化后的对话、token ID、loss_mask存入new_examples
            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function, # 预处理函数
        batched=True, # 启用批处理
        num_proc=num_proc, # 并行进程数
        remove_columns=original_columns1, # 删除原始列，如id等原始字段
        load_from_cache_file=False # 强制重新运行预处理，即使之前已执行过map()处理
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
ds = build_dataset_rank(bigtokenizer)
print(ds) 

# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)

bigmodel.eval()

@torch.no_grad() # 关闭梯度计算以节省显存
def ge(data):
    input_ids=data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True) # 计算模型输出
    hidden_state_big = outs_big.hidden_states[-1] # 提取最后一层隐藏状态
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1) # 计算softmax概率
    maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# 将ge(data)计算的结果保存到.ckpt文件
def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

# 主循环，遍历数据集，每100次打印一次日志，运行ge()并保存结果
for id,data in enumerate(ds):
    if id%100==0:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir,outdata)


