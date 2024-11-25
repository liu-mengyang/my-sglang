import json

import numpy as np


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
    with open(f"{save_name}_logits.jsonl", 'a') as f:
        logits_dict["Inp"] = np.array(logits_dict["Inp"].cpu())
        logits_dict["Out"] = np.array(logits_dict["Out"].cpu())
        json.dump(logits_dict, f, cls=NpEncoder)
        f.write("\n")


def save_results(results_dict, save_name):
    with open(f"{save_name}_results.jsonl", 'a') as f:
        json.dump(results_dict, f, cls=NpEncoder)
        f.write("\n")
