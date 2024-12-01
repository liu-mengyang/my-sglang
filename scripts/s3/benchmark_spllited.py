import os
import time

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
    dataset = load_dataset("/models/cnn_dailymail", split="test")

    runtime = sgl.Runtime(model_path="/models/Mixtral-8x7B-Instruct-v0.1-splitted",
                          disable_overlap_schedule=True,
                          tp_size=1,
                          disable_cuda_graph=True)
    sgl.set_default_backend(runtime)
    
    for id, article in tqdm(enumerate(dataset["document"][:100])):
        # print(f"Summarizing Article {id}")
        serving_tic = time.time()
        state = summarize.run(article)
        serving_toc = time.time()
        input_text = None
        output_text = None
        for m in state.messages():
            # print(m["role"], ":", m["content"])
            if m["role"] == "user":
                input_text = m["content"]
            elif m["role"] == "assistant":
                output_text = m["content"]
        assert input_text is not None and output_text is not None
        # clear buffer
        buffer = []

        print(f"Serving with {serving_toc - serving_tic} seconds")
        # print(state["result"])


if __name__ == "__main__":
    main()