import torch
# from unsloth import FastVisionModel
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from dataset.load import DatasetLoader

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