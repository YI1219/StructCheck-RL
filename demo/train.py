import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from unsloth import FastLanguageModel

# ---------------------------------------------------------
# 1. Load MiniCPM-V-4_5 (quantized) using Unsloth
# ---------------------------------------------------------
model_name = "openbmb/MiniCPM-V-4_5"

print("Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,          # quantized loading
    max_seq_length=2048,
    dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# 2. Load a tiny demo dataset (HuggingFace)
# ---------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("imdb", split="train[:1%]")  # small subset for demo

def format_prompt(example):
    return f"Review: {example['text']}\n\nSentiment?"

dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})

# ---------------------------------------------------------
# 3. Define a simple GRPO-style reward function
#    (positive sentiment → reward 1, negative → reward 0)
# ---------------------------------------------------------
positive_words = ["good", "great", "excellent", "amazing", "love"]
negative_words = ["bad", "terrible", "awful", "hate", "poor"]

def compute_reward(text):
    text = text.lower()
    score = 0
    if any(w in text for w in positive_words):
        score += 1
    if any(w in text for w in negative_words):
        score -= 1
    return float(score > 0)

# ---------------------------------------------------------
# 4. PPO Configuration
# ---------------------------------------------------------
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# ---------------------------------------------------------
# 5. PPO Training Loop
# ---------------------------------------------------------
print("Starting PPO training...")

for batch in ppo_trainer.dataloader:
    prompts = batch["prompt"]

    # Generate model responses
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
    )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Compute rewards
    rewards = [compute_reward(r) for r in responses]

    # PPO step
    ppo_trainer.step(prompts, responses, rewards)

    print("\n--- Example ---")
    print("Prompt:", prompts[0])
    print("Response:", responses[0])
    print("Reward:", rewards[0])
    print("----------------\n")

print("Training complete!")
