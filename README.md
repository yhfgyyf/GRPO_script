# GRPO_script
一些grpo训练脚本，包括奖励函数

环境： ubuntu 22.04 LTS cuda==12.4 torch==2.5.1 vllm==0.7.2 trl==0.14.0

单卡48g显存，显存消耗45g左右，模型qwen2.5-1.5b-instruct 使用3B模型需要单卡80g显存，实际消耗72g左右

grpo_bley.py 用于训练粤语，并按照指定格式[ASSISTANT]输出
