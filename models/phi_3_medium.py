import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_phi():
    model_id = "microsoft/Phi-3-medium-128k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def prompt_template(prompt):
        template = [
            {"role": "user", "content": prompt},
        ]
        return template
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128000,
        temperature=1.0,
        return_full_text = False
    )

    return model, tokenizer, pipe, prompt_template