import os

from datasets import load_dataset, load_from_disk
import sglang as sgl
from sglang import function, system, user, assistant, gen
from sglang.srt.managers.s3_utils import save_results
from tqdm import tqdm

hf_url = "https://huggingface.co/datasets/cais/mmlu"
os.environ["S3_DATASET_NAME"] = "mmlu"


@function
def qa(s, prompt):
    s += system("You are a knowledge expert, you are supposed to answer the \
        multi-choice question to derive your final answer as \
        'The answer is...'.")
    s += user(prompt)
    s += assistant(gen("question", max_tokens=256))


def main():
    local_dataset_path = "./datasets/"+ os.getenv("S3_DATASET_NAME", "other")
    try:
        dataset = load_from_disk(local_dataset_path)
    except:
        print("Local dataset not found, downloading from HuggingFace...")
        dataset = load_dataset("cais/mmlu", 'all')
        os.makedirs("./datasets", exist_ok=True)
        dataset.save_to_disk(local_dataset_path)
        print(f"Dataset saved to {local_dataset_path}")
    
    runtime = sgl.Runtime(model_path="/models/Mixtral-8x7B-Instruct-v0.1",
                          disable_overlap_schedule=True,
                          tp_size=8,
                          disable_cuda_graph=True)
    sgl.set_default_backend(runtime)
    
    for id, question in tqdm(enumerate(dataset["test"]["question"])):
        state = qa.run(question)
        input_text = None
        output_text = None
        for m in state.messages():
            # print(m["role"], ":", m["content"])
            if m["role"] == "user":
                input_text = m["content"]
            elif m["role"] == "assistant":
                output_text = m["content"]
        assert input_text is not None and output_text is not None
        save_results({
            "Input_text": input_text,
            "Output_text": output_text}, os.getenv("S3_DATASET_NAME"))
    # print(state["result"])


if __name__ == "__main__":
    main()