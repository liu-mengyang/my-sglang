import os

from datasets import load_dataset, load_from_disk
import sglang as sgl
from sglang import function, system, user, assistant, gen
from sglang.srt.managers.s3_utils import save_results
from tqdm import tqdm

hf_url = "https://huggingface.co/datasets/abisee/cnn_dailymail"
os.environ["S3_DATASET_NAME"] = "cnn_dailymail"


@function
def summarize(s, article):
    s += system("Your task is to summarize the user given article in only one \
                 paragraph.")
    s += user(article)
    s += assistant(gen("summarization", max_tokens=256))


def main():
    local_dataset_path = "./datasets/"+ os.getenv("S3_DATASET_NAME", "other")
    try:
        dataset = load_from_disk(local_dataset_path)
    except:
        print("Local dataset not found, downloading from HuggingFace...")
        splits = ['1.0.0', '2.0.0', '3.0.0'] #prefer to use 2.0.0
        os.makedirs("./datasets", exist_ok=True)
        for i, split_name in enumerate(splits):
            split_path = os.path.join(local_dataset_path, split_name)
            os.makedirs(split_path, exist_ok=True)
            dataset = load_dataset("abisee/cnn_dailymail", split_name, trust_remote_code=True)
            dataset.save_to_disk(split_path)
        print(f"Dataset saved to {local_dataset_path}")
        
    runtime = sgl.Runtime(model_path="/models/Mixtral-8x7B-Instruct-v0.1",
                          disable_overlap_schedule=True,
                          tp_size=8,
                          disable_cuda_graph=True)
    sgl.set_default_backend(runtime)
    for id, article in tqdm(enumerate(dataset["document"])):
        # print(f"Summarizing Article {id}")
        state = summarize.run(article)
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
            "Output_text": output_text}, "test")
    # print(state["result"])


if __name__ == "__main__":
    main()