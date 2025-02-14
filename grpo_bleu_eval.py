import re
from datasets import load_dataset, Dataset
from grpo_bleu_4 import calculate_bleu4
from vllm import LLM,SamplingParams
import json

SYSTEM_PROMPT = """
请严格按照以下格式回答:
[ASSISTANT]
<回答内容>
[/ASSISTANT]
"""


def get_from_local(json_path: str) -> Dataset:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_dict({
        'question': [item['question'] for item in data],
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
    
datasets=get_from_local("grpo_bleu_test.json")

total=len(datasets)


model_path="/home/hfcloudy/outputs/GRPO_bleu_real_model/checkpoint-100"
model=LLM(model=model_path,max_model_len=16384,gpu_memory_utilization=0.4,enforce_eager=True)
# tokenizer = AutoTokenizer(model_path)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
score_list=[]
i=0
for data in datasets:
    i+=1
    question=data["question"]
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role" : "user", "content" : question}]
    outputs = model.chat(
    messages=messages,
    sampling_params=sampling_params,
    )

    for o in outputs:
        response = o.outputs[0].text
    res = response.replace("[/ASSISTANT]",'')
    
    bleu4_score=calculate_bleu4(res,data["answer"])
    score_list.append(bleu4_score)
    print(f"{i+1} bleu4_score: {bleu4_score}")
    
        
acc = round(sum(score_list)/total, 4)
print(acc)
