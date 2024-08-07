import os
import psutil
import torch
import GPUtil
from datetime import datetime
import matplotlib.pyplot as plt

def log_memory_usage(step="", log_file=None):
    process = psutil.Process()
    memory_info = f"{step} - Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB"
    gpu_memory_info = f"{step} - GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB" if torch.cuda.is_available() else "No GPU available"
    gpus = GPUtil.getGPUs()
    gpu_usage_info = f"{step} - GPU Usage: {gpus[0].load * 100 if gpus else 'N/A'} %"

    log_message = f"{memory_info}\n{gpu_memory_info}\n{gpu_usage_info}\n"
    print(log_message)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_message + "\n")

def save_plot(fig, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")

def create_output_dir(base_output_dir):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
