import unsloth  # must be imported first
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen2.5-1.5B-Instruct"

print("Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    max_seq_length=1024,
    dtype=torch.float16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

print("Loading dataset...")
dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:200]")

positive_words = ["good", "great", "excellent", "amazing", "love", "helpful", "clear"]
negative_words = ["bad", "terrible", "awful", "hate", "poor", "wrong", "error"]

def reward_fn(completions, **kwargs):
    scores = []
    for completion in completions:
        text = completion[0]["content"].lower() if isinstance(completion, list) else str(completion).lower()
        score = 0.0
        score += 0.5 * sum(w in text for w in positive_words)
        score -= 0.5 * sum(w in text for w in negative_words)
        score += min(len(text) / 200.0, 1.0)
        scores.append(score)
    return scores

training_args = GRPOConfig(
    output_dir="/app/grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_completion_length=128,
    learning_rate=5e-6,
    logging_steps=5,
    save_steps=50,
    fp16=True,
)

print("Initializing GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting GRPO training...")
trainer.train()
print("Training complete!")
