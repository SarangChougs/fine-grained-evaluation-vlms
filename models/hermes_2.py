import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes, flash_attn

def load_hermes_2():
    
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Theta-Llama-3-8B', trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        "NousResearch/Hermes-2-Theta-Llama-3-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=False,
        load_in_4bit=True,
        use_flash_attention_2=True
    )
    
    def prompt_template(prompt):
        schema = '''
        {
          "id": "",
          "input": "Input text",
          "ground_truth": "Ground Truth",
          "Scores": [
            {
              "Metric": "",
              "Comments": "",
              "Value": ""
            }
            ],
          "Total score": "",
          "Other remarks": "",
        }
        '''
        messages = [
            {"role": "system", "content": '''You are a helpful assistant that answers in JSON.'''},
            {"role": "user", "content": prompt}
        ]
        return messages
        
    def run_model(prompt):
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
        generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=1.0, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        return response
    
    return tokenizer, model, run_model, prompt_template

