import openai

from datasets import load_dataset
import sglang as sgl
from sglang import function, system, user, assistant, gen
from sglang.srt.managers.s3_utils import save_results
from tqdm import tqdm


@function
def coding(s, prompt):
    s += system("You are a helpful code assistant that help with writing Python code for a user request.")
    s += user(prompt)
    s += assistant(gen("code", max_tokens=256))


def main():
    dataset = load_dataset("google-research-datasets/mbpp", split='test')
    runtime = sgl.Runtime(model_path="/models/Mixtral-8x7B-Instruct-v0.1",
                          disable_overlap_schedule=True,
                          tp_size=8,
                          disable_cuda_graph=True)
    sgl.set_default_backend(runtime)

    for id, code in tqdm(enumerate(dataset["text"])):
        state = coding.run(code)
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
            "Output_text": output_text}, "mbpp")

    # print(state["result"])

if __name__ == "__main__":
    main()