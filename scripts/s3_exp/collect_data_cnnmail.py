import requests

from datasets import load_dataset
from tqdm import tqdm


dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
for input_text in tqdm(dataset["article"]):
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": f"{input_text}",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 2048,
            },
        },
    )
