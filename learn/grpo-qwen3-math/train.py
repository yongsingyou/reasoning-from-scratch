import subprocess
import sys

try:
    import google.colab  # only runs on Colab
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
    subprocess.check_call(["uv", "pip", "install", "--system", "-q",
        "trl", "transformers", "accelerate", "peft", "datasets", "wandb",
        "torchao>=0.16.0",
    ])
except ImportError:
    pass  # local: dependencies already installed

import json
import os
import re
import urllib.request
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Secrets ───────────────────────────────────────────────────────────────────

from huggingface_hub import login

try:
    from google.colab import userdata
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    hf_token = userdata.get("HF_TOKEN")
    login(hf_token)
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = HERE
TRAIN_PATH = HERE.parent / "data" / "math_train.json"
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

is_cuda = device == "cuda"
print(f"device: {device}, dtype: {dtype}")
if is_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Data ──────────────────────────────────────────────────────────────────────

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
if not TRAIN_PATH.exists():
    print("Downloading math_train.json ...")
    url = "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json"
    urllib.request.urlretrieve(url, str(TRAIN_PATH))
    print(f"Saved to {TRAIN_PATH} ({TRAIN_PATH.stat().st_size / 1e6:.1f} MB)")

with open(TRAIN_PATH) as f:
    data = json.load(f)


def make_prompt(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


subset = data if is_cuda else data[:64]
dataset = Dataset.from_list([
    {"prompt": make_prompt(ex["problem"]), "answer": ex["answer"]}
    for ex in subset
])
print(f"Training dataset size: {len(dataset)} ({'full' if is_cuda else 'local subset'})")

# ── Model ─────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if is_cuda:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

print(f"Model on {device}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

# ── Rewards ───────────────────────────────────────────────────────────────────

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def reward_correctness(prompts, completions, answer, **kwargs):
    scores = []
    for completion, gt in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        extracted = extract_boxed(text)
        scores.append(1.0 if extracted is not None and extracted == gt.strip() else 0.0)
    return scores


def reward_format(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        scores.append(0.5 if has_think else 0.0)
    return scores

# ── Training ──────────────────────────────────────────────────────────────────

wandb.init(
    project="grpo-qwen3-math",
    config={
        "model": MODEL_ID,
        "device": device,
        "num_generations": 4 if is_cuda else 2,
        "max_completion_length": 512 if is_cuda else 128,
        "learning_rate": 1e-6,
        "beta": 0.01,
    },
)

config = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    num_generations=4 if is_cuda else 2,
    max_completion_length=512 if is_cuda else 128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8 if is_cuda else 2,
    learning_rate=1e-6,
    num_train_epochs=1,
    max_steps=-1 if is_cuda else 10,
    beta=0.01,
    bf16=is_cuda,
    fp16=False,
    logging_steps=1,
    save_steps=50 if is_cuda else 5,
    save_total_limit=3,
    save_only_model=True,
    report_to="wandb",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    reward_funcs=[reward_correctness, reward_format],
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(str(OUTPUT_DIR / "final"))
print(f"Model saved to {OUTPUT_DIR / 'final'}")

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import whoami
    username = whoami()["name"]
    repo_id = f"{username}/grpo-qwen3-math"
    trainer.push_to_hub(repo_id)
    print(f"Model pushed to hub: {repo_id}")
else:
    print("HF_TOKEN not set — skipping push to hub.")

wandb.finish()
