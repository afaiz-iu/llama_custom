import pandas as pd
import time
from contextlib import contextmanager
import torch
from generation import Llama
from model import ModelArgs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch", message="torch.set_default_tensor_type")

def main():
    e_s = [512, 768, 1024, 1536, 2048, 4096] # embedding sizes
    r = range(2,43,2) # num heads (e_s should be divisible by num_heads)
    l_s = range(8,33) 
    d = {}

    for _ in r:
        d[f"{_}"] = []
        d[f"{_}"].extend([emb_sz for emb_sz in e_s if emb_sz%_ == 0])

    d = {k:v for k,v in d.items() if v}

    num_layers_range = range(25, 33)
    batch_size_range = range(1, 6)

    results = []

    for num_heads, embedding_sizes in d.items():
        for embedding_size in embedding_sizes:
            for num_layers in num_layers_range:
                for batch_size in batch_size_range:
                    model_args = ModelArgs(
                        dim=embedding_size,
                        n_layers=num_layers,
                        n_heads=int(num_heads),
                        max_seq_len=2048,
                        max_batch_size=batch_size
                    )

                    try:
                        start_time = time.time()
                        model = Llama.build(
                            tokenizer_path='tokenizer.model',
                            max_seq_len=2048,
                            max_batch_size=batch_size,
                            model_args=model_args,
                            seed=99
                        )
                        build_time = time.time() - start_time
                        num_p = sum(p.numel() for p in model.model.parameters())
                        oom = "no"
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            oom = "yes"
                            build_time = None
                            num_p = None
                        else:
                            raise e

                    results.append({
                    "batch_size": batch_size,
                    "embedding_size": embedding_size,
                    "num_heads": int(num_heads),
                    "num_layers": num_layers,
                    "total_num_parameters": num_p,
                    "OOM": oom,
                    "build_time(ms)": build_time * 1000 if build_time else None
                    })

                # results.append({
                #     "batch_size": batch_size,
                #     "embedding_size": embedding_size,
                #     "num_heads": int(num_heads),
                #     "num_layers": num_layers,
                #     "total_num_parameters": num_p,
                #     "OOM": oom,
                #     "build_time(ms)": build_time * 1000 if build_time else None
                # })

    df = pd.DataFrame(results)
    df.to_csv("load_times.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()