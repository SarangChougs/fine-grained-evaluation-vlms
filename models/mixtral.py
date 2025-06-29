from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_mixtral():
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", use_fast = True)
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", device_map="auto")
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=750,
    model_kwargs={"torch_dtype": torch.bfloat16},
    return_full_text = False
    )
    def prompt_template(prompt):
        return f'''[INST] {prompt} [/INST]'''

    return tokenizer, model, pipe, prompt_template