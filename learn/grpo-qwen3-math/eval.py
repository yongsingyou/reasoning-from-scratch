import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

# ── Config ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = HERE / "final"
EVAL_PATH = HERE.parent / "data" / "math500_test.json"
N_EVAL = 500 if torch.cuda.is_available() else 5
SYSTEM_PROMPT = (
    "You are a math expert. Show your reasoning inside <think>...</think> tags, "
    "then give the final answer inside \\boxed{}."
)

# ── Device ────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"device: {device}, dtype: {dtype}")

# ── Data ──────────────────────────────────────────────────────────────────────

EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
if not EVAL_PATH.exists():
    print("Downloading math500_test.json ...")
    url = "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/main/ch03/01_main-chapter-code/math500_test.json"
    import urllib.request
    urllib.request.urlretrieve(url, str(EVAL_PATH))
    print(f"Saved to {EVAL_PATH}")

with open(EVAL_PATH) as f:
    data = json.load(f)


def make_prompt(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


# ── Model ─────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print(f"Loaded adapter from {ADAPTER_DIR}")

# ── Eval ──────────────────────────────────────────────────────────────────────

if not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = os.environ.get("wandb_apikey", "")

wandb.init(
    project="grpo-qwen3-math",
    job_type="eval",
    config={
        "adapter": ADAPTER_DIR,
        "n_eval": N_EVAL,
    },
)

# ── Sanity check: print first example before full eval ────────────────────────

def generate(ex):
    encoded = tokenizer.apply_chat_template(
        make_prompt(ex["problem"]),
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=256, temperature=0.0, do_sample=False,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

sample = data[0]
sample_completion = generate(sample)
print(f"Problem : {sample['problem']}")
print(f"Response: {sample_completion}")
print(f"Expected: {sample['answer']}")
print(f"Extracted: {extract_boxed(sample_completion)}")
print("-" * 60)

correct = 0
for ex in data[:N_EVAL]:
    completion = generate(ex)
    extracted = extract_boxed(completion)
    if extracted and extracted == ex["answer"].strip():
        correct += 1

accuracy = correct / N_EVAL
wandb.log({"eval/accuracy": accuracy, "eval/n": N_EVAL})
wandb.finish()

print(f"Accuracy: {correct}/{N_EVAL} = {accuracy:.1%}")
