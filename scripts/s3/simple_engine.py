import openai

from datasets import load_dataset
import sglang as sgl
from sglang import function, system, user, assistant, gen
from sglang.srt.managers.s3_utils import save_results


@function
def summarize(s, article):
    s += system("Your task is to summarize the user given article in only one \
                 paragraph.")
    s += user(article)
    s += assistant(gen("summarization", max_tokens=256))


def main():
    dataset = load_dataset("/models/cnn_dailymail", split="test")
    runtime = sgl.Runtime(model_path="/models/Meta-Llama-3.1-8B-Instruct",
                          disable_overlap_schedule=True)
    sgl.set_default_backend(runtime)

    for id, article in enumerate(dataset["article"][:3]):
        print(f"Summarizing Article {id}")
        state = summarize.run(article)
        input_text = None
        output_text = None
        for m in state.messages():
            print(m["role"], ":", m["content"])
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