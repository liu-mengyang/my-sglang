import json
import os

import numpy as np
from safetensors.torch import save_file
import torch


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_logits(logits_dict, save_name):
    # Inp
    output_path = f"outputs/{save_name}_{logits_dict['Req_id']}_{logits_dict['Inp_id']}.safetensors"
    os.makedirs(output_path, exist_ok=True)
    save_dict = {}
    save_dict["Inp"] = logits_dict["Inp"].cpu()
    save_dict["Out"] = logits_dict["Out"].cpu()
    save_dict["Score"] = torch.tensor(logits_dict["scores"])
    save_dict["Activation"] = torch.tensor(logits_dict["activation"])
    save_file(save_dict, output_path)


def save_results(results_dict, save_name):
    with open(f"outputs/{save_name}_results.jsonl", 'a') as f:
        json.dump(results_dict, f, cls=NpEncoder)
        f.write("\n")
