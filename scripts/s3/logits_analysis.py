import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from safetensors.torch import save_file


# build a vocab dictionary
def build_vocab(vocab_dict, data, cnt):
    for i in range(len(data['Inp'])):
        for j in range(len(data['Inp'][i])):
            if data['Inp'][i][j] not in vocab_dict:
                vocab_dict[data['Inp'][i][j]] = cnt
                cnt += 1
    return vocab_dict, cnt


# collect expert activation
def collect_activation(expert_cnt, data, vocab_dict):
    for i in range(len(data['Inp'])):
        for j in range(len(data['Inp'][i])):
            token_id = vocab_dict[data['Inp'][i][j]]
            for m in range(32):
                activation = data['activation'][i][m][j]
                for n in range(8):
                    e_decision = np.array(activation) == n
                    # print(activation)
                    e_result = e_decision.sum()
                    # print(e_result)
                    expert_cnt[token_id,m,n] += e_result
    return expert_cnt


results_file = "/hkust/my-sglang/test_results.jsonl"
logits_file = "/hkust/my-sglang/test_logits.jsonl"
tmp_file = "/hkust/my-sglang/temp.jsonl"
save_vocab_file = "/hkust/my-sglang/vocab.json"
save_expert_file = "/hkust/my-sglang/experts.npy"

lines = open(logits_file).readlines()

vocab_dict = {}

chunk_size = 100
cnt = 0
# build vocab dict
for i in tqdm(range(len(lines) // chunk_size + 1)[:1]):
    st_flag = False
    idx = (i+1)*chunk_size
    if idx > len(lines):
        idx = len(lines)
    for line in lines[i*chunk_size:idx]:
        open(tmp_file, 'a' if st_flag else 'w').write(line)
        st_flag = True
    logits = pd.read_json(path_or_buf=open(tmp_file), lines=True)

    vocab_dict, cnt = build_vocab(vocab_dict, logits, cnt)

json.dump(vocab_dict, open(save_vocab_file, 'w'))

expert_cnt = np.zeros((len(vocab_dict),32,8))
# collect activation
for i in tqdm(range(len(lines) // chunk_size + 1)[:1]):
    st_flg = False
    idx = (i+1)*chunk_size
    if idx > len(lines):
        idx = len(lines)
    for line in lines[i*chunk_size:idx]:
        open(tmp_file, 'a' if st_flag else 'w').write(line)
        st_flag = True
    logits = pd.read_json(path_or_buf=open(tmp_file), lines=True)

    expert_cnt = collect_activation(expert_cnt, logits, vocab_dict)

os.makedirs('outputs', exist_ok=True)
save_file({"act":torch.tensor(expert_cnt)}, 'outputs/test.safetensors')

# np.save(open(save_expert_file, 'wb'), expert_cnt)