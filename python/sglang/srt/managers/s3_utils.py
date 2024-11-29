import json
import os

import numpy as np
from safetensors.torch import save_file
import torch

import sqlite3
import pickle
# for initializing the database
def init_logits_db(db_name="s3_logits.db"):
    """Initialize SQLite database for logits"""
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            save_name TEXT,
            req_id TEXT,
            inp_id TEXT,
            inp_tensor BLOB,
            out_tensor BLOB,
            scores BLOB,
            activation BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_logits_tofile(logits_dict, save_name):
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


# for saving logits to the database
def save_logits(logits_dict, save_name):
    """Save logits to SQLite database"""
    conn = init_logits_db()
    cursor = conn.cursor()
    
    # Convert tensors to bytes using pickle
    inp_bytes = pickle.dumps(logits_dict["Inp"].cpu())
    out_bytes = pickle.dumps(logits_dict["Out"].cpu())
    scores_bytes = pickle.dumps(torch.tensor(logits_dict["scores"]))
    activation_bytes = pickle.dumps(torch.tensor(logits_dict["activation"]))
    
    cursor.execute('''
        INSERT INTO logits 
        (save_name, req_id, inp_id, inp_tensor, out_tensor, scores, activation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        save_name,
        str(logits_dict['Req_id']),
        str(logits_dict['Inp_id']),
        inp_bytes,
        out_bytes,
        scores_bytes,
        activation_bytes
    ))
    
    conn.commit()
    conn.close()


# for loading logits from the database
def load_logits_from_data(db_name="s3_logits.db", req_id=None):
    """Load scores and activation from SQLite"""
    conn = init_logits_db(db_name)
    cursor = conn.cursor()
    
    query = "SELECT save_name, req_id, inp_id, scores, activation FROM logits"
    params = []
    if req_id:
        query += " WHERE req_id = ?"
        params.append(req_id)
    
    cursor.execute(query, params)
    results = []
    
    for row in cursor.fetchall():
        result = {
            "save_name": row[0],
            "req_id": row[1],
            "inp_id": row[2],
            "scores": torch.tensor(pickle.loads(row[3])),
            "activation": torch.tensor(pickle.loads(row[4]))
        }
        results.append(result)
    
    conn.close()
    return results