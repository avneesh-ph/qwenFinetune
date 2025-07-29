import torch
# from trl import SFTTrainer
# from unsloth import FastVisionModel
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from dataset.load import DatasetLoader
from config.train_config import training_args
from unsloth.trainer import UnslothVisionDataCollator

# Check hardware capabilities
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load Unsloth's 4-bit quantized Qwen 2.5-VL model
# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
#     load_in_4bit=True,
#     use_gradient_checkpointing="unsloth"
# )

# print("Model loaded successfully!")
# print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

datasetLoader = DatasetLoader("ift/handwriting_forms")
# datasetLoader = DatasetLoader("HuggingFaceM4/ChartQA")
formatted_train, formatted_eval = datasetLoader.loadDataset()

# Create the trainer
# trainer = SFTTrainer(
#     model=model,
#     processing_class=tokenizer,
#     data_collator=UnslothVisionDataCollator(model, tokenizer),
#     train_dataset=formatted_train,
#     eval_dataset=formatted_eval,
#     args=training_args,
# )

# # Check memory usage before training
# print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# # Start training
# print("\nStarting training...")
# print("This will take 30-60 minutes depending on your hardware.")
# print("=" * 60)

# trainer.train()

# print("\nTraining completed!")
# print(f"GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")