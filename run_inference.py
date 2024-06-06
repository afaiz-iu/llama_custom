import torch
import torch.nn as nn
from generation import Llama, sample_top_p
from model import ModelArgs
from tokenizer import Tokenizer
import fire
import random
import string
import time
import pandas as pd 
import numpy as np
import os
import logging

from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch", message="torch.set_default_tensor_type")

def print_model_structure(model: nn.Module):
    for name, module in model.named_modules():
        if name:
            print(f"{name}: {module}")

def gen_random_prompt(length):
    characters = string.ascii_letters + string.digits
    prompt = ''.join(random.choice(characters) for _ in range(length))
    return prompt

def setup_logging(log_level):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'inference_profiling.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main(num_runs=8, config_file='load_times.csv', ip_prompt_len=1, op_prompt_len=1024, log_level='INFO'):
    setup_logging(log_level)
    
    torch.manual_seed(0)

    cnfg = pd.read_csv(config_file)
    cnfg = cnfg[cnfg["OOM"] == "no"]

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    for idx, row in cnfg.iterrows():
        batch_size = row['batch_size']
        embedding_size = row['embedding_size']
        num_heads = row['num_heads']
        num_layers = row['num_layers']
        num_p = row['total_num_parameters']

        logging.info(f"Processing row {idx}: batch_size={batch_size}, embedding_size={embedding_size}, num_heads={num_heads}, num_layers={num_layers}, num_p={num_p}")

        prompts = [gen_random_prompt(ip_prompt_len) for _ in range(batch_size)]

        model_args = ModelArgs(
            dim=embedding_size,
            n_layers=num_layers,
            n_heads=num_heads,
            max_seq_len=2048,
            max_batch_size=len(prompts)
        )

        model = Llama.build(
            tokenizer_path='tokenizer.model',
            max_seq_len=2048,
            max_batch_size=len(prompts),
            model_args=model_args,
            seed=99
        )

        inf_times_perf = []
        inf_times_cuda = []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings_seq = np.zeros((num_runs, 1))
        for _ in range(num_runs):
            start = time.perf_counter()
            starter.record()
            completions = model.text_completion(
                prompts,
                max_gen_len=op_prompt_len,
                temperature=0,
                logprobs=False,
            )
            ender.record()        
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings_seq[_] = curr_time            
            e2e_inf_time = (time.perf_counter() - start) * 1000
            inf_times_perf.append(e2e_inf_time)
            inf_times_cuda.append(curr_time)

        mean_inf_time_perf = sum(inf_times_perf) / len(inf_times_perf)
        std_inf_time_perf = (sum((x - mean_inf_time_perf)**2 for x in inf_times_perf) / (len(inf_times_perf) - 1))**0.5

        mean_inf_time_cuda = sum(inf_times_cuda) / len(inf_times_cuda)
        std_inf_time_cuda = (sum((x - mean_inf_time_cuda)**2 for x in inf_times_cuda) / (len(inf_times_cuda) - 1))**0.5

        logging.info(f"Inference times for row {idx}: mean_inf_time_perf={mean_inf_time_perf:.2f}ms, std_inf_time_perf={std_inf_time_perf:.2f}ms, mean_inf_time_cuda={mean_inf_time_cuda:.2f}ms, std_inf_time_cuda={std_inf_time_cuda:.2f}ms")

        result = {
            'input_prompt_len': ip_prompt_len,
            'output_prompt_len': op_prompt_len,
            'batch_size': batch_size,
            'embedding_size': embedding_size,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'num_P': num_p,
            'mean_inf_time_perf': mean_inf_time_perf,
            'std_inf_time_perf': std_inf_time_perf,
            'mean_inf_time_cuda': mean_inf_time_cuda,
            'std_inf_time_cuda': std_inf_time_cuda
        }

        result_df = pd.DataFrame([result])
        result_file = os.path.join(results_dir, f'results_{idx}.csv')
        result_df.to_csv(result_file, index=False)

if __name__ == "__main__":
    fire.Fire(main)
