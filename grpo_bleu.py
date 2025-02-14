###训练目标为输出[ASSISTANT]....

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """
请严格按照以下格式回答:
[ASSISTANT]
<回答内容>
[/ASSISTANT]
"""
import json
def get_from_local(json_path: str) -> Dataset:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_dict({
        'question': [item['question'] for item in data],  # 添加原始问题
        'prompt': [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': item['question']}
            ]
            for item in data
        ],
        'answer': [item['answer'] for item in data]
    })
    
    return dataset

dataset=get_from_local("grpo_test.json")


def assistant_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct [ASSISTANT]...[/ASSISTANT] format."""
    pattern = r"^\[ASSISTANT\](.*?)\[/ASSISTANT\]$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_assistant_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains [ASSISTANT]...[/ASSISTANT] tags."""
    pattern = r"\[ASSISTANT\](.*?)\[/ASSISTANT\]"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_assistant_format(text) -> float:
    count = 0.0
    if text.startswith("[ASSISTANT]"):
        count += 0.25
    else:
        if "[ASSISTANT]" in text:
            count += 0.125
    if text.endswith("[/ASSISTANT]"):
        count += 0.25
    else:
        if "[/ASSISTANT]" in text:
            count += 0.125
    if text.endswith("[/ASSISTANT]"):
        prefix = text.split("[ASSISTANT]")[0]
        count -= len(prefix.strip()) * 0.001
        suffix = text.split("[/ASSISTANT]")[-1]
        count -= len(suffix.strip()) * 0.001
    start_tags = text.count("[ASSISTANT]")
    end_tags = text.count("[/ASSISTANT]")
    if start_tags > 1:
        count -= (start_tags - 1) * 0.1
    if end_tags > 1:
        count -= (end_tags - 1) * 0.1
        
    return max(0.0, count)  

def assistant_format_count_reward(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_assistant_format(c) for c in contents]

from collections import Counter
import math

def get_ngrams(segment, n):
    ngrams = []
    for i in range(len(segment) - n + 1):
        ngram = segment[i:i + n]
        ngrams.append(ngram)
    return ngrams

def count_ngrams(segment, n):
    ngram_counts = Counter()
    for ngram in get_ngrams(segment, n):
        ngram_counts[ngram] += 1
    return ngram_counts

def modified_precision(candidate, reference, n):
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = count_ngrams(reference, n)
    
    if len(candidate_ngrams) == 0:
        return 0
    
    clipped_counts = {}
    for ngram, count in candidate_ngrams.items():
        clipped_counts[ngram] = min(count, reference_ngrams[ngram])
    
    numerator = sum(clipped_counts.values())
    denominator = sum(candidate_ngrams.values())
    
    return numerator / denominator if denominator != 0 else 0

def brevity_penalty(candidate, reference):

    c = len(candidate)
    r = len(reference)
    
    if c > r:
        return 1
    elif c == 0:
        return 0
    else:
        return math.exp(1 - r/c)

def calculate_bleu4(candidate, reference):

    p1 = modified_precision(candidate, reference, 1)
    p2 = modified_precision(candidate, reference, 2)
    p3 = modified_precision(candidate, reference, 3)
    p4 = modified_precision(candidate, reference, 4)

    if 0 in (p1, p2, p3, p4):
        return 0
    avg = math.exp((math.log(p1) + math.log(p2) + math.log(p3) + math.log(p4)) / 4)
    bp = brevity_penalty(candidate, reference)
    return bp * avg

def bleu_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    计算BLEU奖励分数
    Args:
        prompts: 输入提示
        completions: 模型生成的完成
    Returns:
        float: 奖励分数
    """
    responses = []
    for completion in completions:
        if isinstance(completion, list):
            response = completion[0]['content'].replace("[/ASSISTANT]",'')
        else:
            response = completion['content'].replace("[/ASSISTANT]",'')
        if isinstance(response, list):
            response = ' '.join(response)  # 如果response是list，将其连接成字符串
        responses.append(response)

    query = prompts[0][-1]['content']
    if isinstance(query, list):
        query = ' '.join(query)  # 如果query是list，将其连接成字符串

    rewards = []
    
    # 为每个response计算BLEU分数和奖励，从0.4开始线性增加至1.0,公式reward = 1.0 + ((bleu_score - 0.4) / 0.6)
    for response in responses:
        try:
            bleu_score = calculate_bleu4(response, query)
            
            # 根据BLEU分数确定奖励
            if bleu_score < 0.4:
                reward = 0.0
            elif bleu_score == 1.0:
                reward = 2.0
            else:
                reward = 1.0 + ((bleu_score - 0.4) / 0.6)
            
            rewards.append(reward)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            rewards.append(0.0)  # 发生错误时返回0分
    
    return rewards


max_seq_length = 512 
model_path="/home/yyf/data/hub/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token


training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, 
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 1, 
    save_steps = 500,
    max_grad_norm = 0.1,
    report_to = "none", #
    output_dir = "outputs/GRPO_model",
    vllm_device="cuda:0",
    vllm_gpu_memory_utilization=0.3,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        assistant_format_count_reward,
        assistant_format_reward_func,
        soft_assistant_format_reward_func,
        bleu_reward_func,
        
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
