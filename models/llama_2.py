from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
def load_llama_2():
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    def prompt_template(prompt):
        return f'''[INST] <<SYS>>
    You are a fair LLM evaluator. Use the Task Description to evaluate the model response. Stick to the description strictly. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    {prompt}[/INST]'''

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        return_full_text = False
    )
    return tokenizer, model, pipe, prompt_template