from eagle.model.ea_model import EaModel
# EaModel，来自eagle.model.ea_model模块，一个用于NLP任务的模型类，支持从预训练模型加载权重
from fastchat.model import get_conversation_template
# get_conversation_template，来自fastchat.model，用于获取对话模板（如vicuna）
import torch

def warmup(model):
    # 按照目标模型类型创建对话模板
    conv = get_conversation_template(args.model_type)
    if args.model_type == "llama-2-chat":
        # Llama 2 Chat版本需要一个系统提示词，确保其回答安全无偏见符合道德，其他模型可能就无需这种额外约束了
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>" # 特定结束符

    your_message="who are you?"

    # 将用户输入“hello”作为第一个角色（通常是用户）的话加入对话
    conv.append_message(conv.roles[0], your_message)

    # 给第二个角色（通常是AI模型）留一个空的响应位置，等待模型生成
    conv.append_message(conv.roles[1], None)

    # get_prompt()负责将对话格式化成适合EaModel处理的输入文本
    prompt = conv.get_prompt()

    if args.model_type == "llama-2-chat":
        prompt += " "

    # 分词器将prompt转换成token id
    input_ids=model.tokenizer([prompt]).input_ids
    # 再转换成PyTorch张量，并转移到GPU提高推理效率
    input_ids = torch.as_tensor(input_ids).cuda()

    # 进行文本生成
    output_ids = model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512) # eagenerate一次性返回完整的token序列

    output=model.tokenizer.decode(output_ids[0])
    print(output)


# 使用命令行参数，添加参数解析（包已内置于python中）
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea_model_path",
    type=str,
    default="/home/xwwen/EAGLE_test/EAGLE-llama2-chat-7B/",
    help="The path of EAGLE weight. This can be a local folder or a Hugging Face repo ID（<组织名或用户名>/<模型名>）."
)
parser.add_argument(
    "--base_model_path",
    type=str,
    default="/home/xwwen/EAGLE_test/Llama-2-7b-chat-hf/",
    help="path of the original model. a local folder or a Hugging Face repo ID"
)
parser.add_argument(
    "--load_in_8bit",
    action="store_true", # 如果提供该参数，则值为True，否则默认为False
    help="use 8-bit quantization"
)
parser.add_argument(
    "--load_in_4bit",
    action="store_true",
    help="use 4-bit quantization"
)
parser.add_argument(
    "--model_type",
    type=str,
    default="llama-2-chat",
    choices=["llama-2-chat","vicuna","mixtral","llama-3-instruct"]
)
parser.add_argument(
    "--total_token",
    type=int,
    default=-1,
    help=" the number of draft tokens"
)
parser.add_argument(
    "--max_new_token",
    type=int,
    default=512,
    help="the maximum number of new generated tokens",
)
args = parser.parse_args()

model = EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    total_token=args.total_token,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    device_map="auto",
)

# 让模型进入推理模式，防止dropout等影响推理
model.eval()

warmup(model)