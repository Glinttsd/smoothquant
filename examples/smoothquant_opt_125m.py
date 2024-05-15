import os,sys
BASE_DIR = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
sys.path.append(BASE_DIR)
import torch
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, quantize_opt


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    
from datasets import load_dataset
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
tokenizer = GPT2Tokenizer.from_pretrained("/data/shidi/LLM_models/facebook/opt-13b")
dataset = load_dataset("lambada", split="validation[:1000]")
evaluator = Evaluator(dataset, tokenizer, "cuda")

model = OPTForCausalLM.from_pretrained(
    "/data/shidi/LLM_models/facebook/opt-125m", torch_dtype=torch.float16, device_map="auto"
)
act_scales = torch.load("/home/shidi/smoothquant/act_scales/opt-125m.pt")
smooth_lm(model, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_opt(model)
# print(model_smoothquant_w8a8)

acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
print(f"SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}")