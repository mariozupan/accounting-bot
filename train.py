#from unsloth import standardize_sharegpt
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel
#from unsloth import FastModel
import torch
import matplotlib.pyplot as plt
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from transformers import TextStreamer




max_seq_length = 1024  
dtype = (
    None  
)
load_in_4bit = True 
load_in_8bit = False
#full_finetuning = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_map = "auto"  


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit, 
    load_in_8bit = load_in_8bit, 
    #full_finetuning = full_finetuning, 
    device_map=device_map,  
)



model = FastLanguageModel.get_peft_model(
    model,
    #Lower r means fewer trainable parameters, which helps retain the base model knowledge
    r= 256, #64, #8, #4, #16, # THis is lora rank  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
   
    lora_alpha = 512,#128, #16, #8, #alpha=2xr #16, #lower, reducing it helps balance new learning and retention.
    lora_dropout=0.05, #0.1, # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True, #"unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)




custom_template = """{%- if tools %}
<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}
{%- else %}
You are Mario, an expert in Croatian double-entry bookkeeping. You provide accurate bookkeeping posting schemes in JSON format respecting the double-entry rule.
{%- endif %}
`
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{%- else %}
{%- if messages[0]['role'] == 'system' %}
<|im_start|>system
{{ messages[0]['content'] }}<|im_end|>
{%- else %}
<|im_start|>system
You are Mario, an expert in Croatian double-entry bookkeeping. You provide accurate bookkeeping posting schemes in JSON format respecting the double-entry rule.
<|im_end|>
{%- endif %}
{%- endif %}
{%- for message in messages %}
{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" %}
<|im_start|>{{ message.role }}
{%- if message.content %}
{{ message.content }}
{%- endif %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}
{%- set tool_call = tool_call.function %}
{%- endif %}
<tool_call>
{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
</tool_call>
{%- endfor %}
<|im_end|>
{%- elif message.role == "tool" %}
{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
<|im_start|>user
{%- endif %}
<tool_response>
{{ message.content }}
</tool_response>
{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
<|im_end|>
{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
"""

custom_eos_token = "eos_token"



tokenizer = get_chat_template(
    tokenizer,
    chat_template=  (custom_template, custom_eos_token),
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token=True    
)

#Loading data from huggingface
#The model was trained on proprietary data, while this is the masked version
from datasets import load_dataset

#load from huggingface
dataset = load_dataset("mariozupan/bookkeeping-posting-schemes-2007-2023")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

#Tokenize the dataset
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print(dataset["train"].features)
print("tokenzized features")
dataset_train = dataset["train"]
dataset_test = dataset["test"]

# To confirm the sizes
print(f"Training set size: {len(dataset_train)}")
print(f"Testing set size: {len(dataset_test)}")

print(dataset["train"].features)  # Check column types
print(dataset["train"][0])        # Verify first example
print(dataset["train"][0].keys())  # Verify columns exist

print("-------The conversation looks like:---------")
print(dataset_train[5]["conversations"])
print("-------The text looks like:---------")
print(dataset_train[5]["text"])


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset=dataset_test,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1, #1,
        gradient_accumulation_steps = 8, #4, # Fixed major bug in latest Unsloth
        warmup_steps = 470, #400, #120, #200, #150,
        #num_train_epochs = 2, # Set this for 1 full training run.
        max_steps =  4700, #4000, #1600, #2000, #1500,
        learning_rate = 1e-5, #1e-4, #2e-4,  #2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 30, #when to start reporting loss, this is just increased for cleaner logs
        optim = "adamw_torch", #"paged_adamw_32bit", # "paged_adamw_8bit", 
        weight_decay = 0.01, #reducing overfitting
        lr_scheduler_type = 'cosine', #"linear", #cosine
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        eval_strategy="steps",  # Evaluate every eval_steps
        eval_steps=30,  # Evaluate every 10 steps        
        ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)




print("verify masking is actually done:")
tokenizer.decode(trainer.train_dataset[5]["input_ids"])


space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
#We can see the System and Instruction prompts are successfully masked!




# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")



trainer_stats = trainer.train()




# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")




# Extract training and evaluation logs
train_loss = []
eval_loss = []
steps = []

for log in trainer.state.log_history:
    if "loss" in log:
        train_loss.append(log["loss"])
        steps.append(log["step"])
    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])

# Plot the losses
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(steps[:len(train_loss)], train_loss, label="Training Loss", marker="o")

# Plot evaluation loss
eval_steps = steps[::len(steps) // len(eval_loss)][:len(eval_loss)]  # Align eval steps
plt.plot(eval_steps, eval_loss, label="Evaluation Loss", marker="x")

# Add labels and title
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)

# Show the plot
#plt.show()
# Save the plot as an image file (PNG, JPG, PDF, etc.)
plt.savefig("loss-7B-cosine-4700-lora64-lora1024-adamw_torch.png", dpi=300, bbox_inches="tight")  # Save as PNG
plt.savefig("loss-7B-cosine-4700-lora64-lora1024-adamw_torch.jpg", dpi=300, bbox_inches="tight")  # Save as JPG





# ### Inference



tokenizer = get_chat_template(
    tokenizer,
    chat_template=  (custom_template, custom_eos_token),
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token=True    
)


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

prompt = "Provide me croatian double-entry bookkeeping posting scheme for INVOICESB, for the year 2020"

# CoT
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

# TIR
messages = [
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]



inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,
                         temperature = 0.7, min_p = 0.9)
tokenizer.batch_decode(outputs)


# ## You can also use a TextStreamer for continuous inference - so you can see the generation token by token, instead of waiting the whole time!


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

prompt = "Provide me croatian double-entry bookkeeping posting scheme for RETOUTPUT"

#prompt = "Document codename for posting invoices for retail goods sold and for moving goods from retail stock."

system_message = """
Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
"""

'''
system_message = """
You are Mario, an expert for Croatian double-entry bookkeeping posting scheme.
If the user asks for the definition of some document codename, respond with the document explanation only, starting with 'In Synesis AIS the codename is used for...'.
If the user provides you some document explanation, respond starting with 'Based on your explanation, Synesis AIS codename is...'.
Do not give a double-entry posting example until the user asks for 'Provide me Croatian double-entry posting scheme for...'.
Consider the question as 'general question' if the user's query didn't start with 'Provide me Croatian double-entry posting scheme for...'.
If the user asks some 'general question', respond with the general answer or demand further information.
If you got a tip, be grateful.
Put the final answer within \\boxed{}.
"""
'''

# CoT
messages = [
    {"role": "system", "content": system_message}, #"Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

# TIR
messages = [
    {"role": "system", "content": system_message},#"Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 560,
                   use_cache = True, temperature = 0.1)




prompt = "Provide the double-entry bookkeeping posting scheme for the document codename 'URAEU'"

system_message = """
You are an expert in accounting and double-entry bookkeeping systems. For each user query regarding a document codename, provide a structured and detailed explanation in JSON format. Your response should include:
1. The year of the example.
2. The document codename.
3. A clear explanation of the document codename's purpose.
4. The total debit and credit values.
5. A detailed list of bookkeeping entries, each specifying:
   - Account number ('ACCOUNT'),
   - Account title (TITLE'),
   - Debit amount ('DEBIT'),
   - Credit amount ('CREDIT').

Ensure the response is precise, formatted as valid JSON, and strictly adheres to the structure provided in the examples. Do not include any extra information outside the specified format.
"""


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024,
                   use_cache = True, temperature = 0.2)



# ## Saving

#model.save_pretrained_gguf("qwencoder7B-1024-lora256-512-cosine-4700steps-adamw_torch-lr1e-5-16bit", tokenizer, quantization_method="f16")

model.save_pretrained_gguf("qwencoder7B-accounting-bot", tokenizer, quantization_method="f16")
#print("The model has been saved")

