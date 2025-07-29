import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.trainer import UnslothVisionDataCollator

# Set up training parameters
training_args = TrainingArguments(
    # Core settings
    per_device_train_batch_size=1,        # Small batches for memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,       # Effective batch size of 16
    warmup_steps=50,                      # Gradual learning rate increase
    num_train_epochs=2,                   # Train for 2 full passes
    max_steps=500,                        # Or stop at 500 steps
    
    # Learning settings
    learning_rate=1e-4,                   # Conservative learning rate
    optim="adamw_8bit",                   # Memory-efficient optimizer
    weight_decay=0.01,                    # Prevent overfitting
    lr_scheduler_type="cosine",           # Smooth learning rate decay
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=50,                        # Evaluate every 50 steps
    save_steps=100,                       # Save checkpoint every 100 steps
    logging_steps=10,                     # Log every 10 steps
    
    # Memory optimization
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    
    # Output settings
    output_dir="./qwen-multitask-form",
    seed=3407,
    data_seed=3407,
)

print("Training configuration ready!")