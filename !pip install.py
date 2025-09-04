!pip install pip3-autoremove 
!pip-autoremove torch torchvision torchaudio -y 
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 
!pip install unsloth 
from unsloth import FastVisionModel 
import torch 
 
model, tokenizer = FastVisionModel.from_pretrained( 
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", 
    load_in_4bit = True, 
    use_gradient_checkpointing = "unsloth", 
) 
 
model = FastVisionModel.get_peft_model( 
    model, 
    finetune_vision_layers     = False, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 
 
    r = 16, 
    lora_alpha = 16, 
    lora_dropout = 0, 
    bias = "none", 
    random_state = 3407, 
    use_rslora = False,  
    loftq_config = None, 
) 
from datasets import load_dataset 
dataset = load_dataset("Santhosh1705kumar/radiology-reports-chest", split = "train") 
dataset 
from PIL import Image 
from io import BytesIO 
 
def load_image_from_dict(image_dict): 
    """ 
    Converts a dictionary containing image bytes into a PIL Image. 
 
    Args: 
        image_dict (dict): A dictionary with a 'bytes' key containing image data. 
 
    Returns: 
        PIL.Image.Image: The decoded image. 
    """ 
    image_bytes = image_dict["bytes"] 
    return Image.open(BytesIO(image_bytes)) 
for i in range(100, 120): 
  image = load_image_from_dict(dataset[i]["image"]) 
  display(image) 
dataset[0]["caption"] 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
def convert_to_conversation(sample): 
    conversation = [ 
        { "role": "user", 
          "content" : [ 
            {"type" : "text",  "text"  : instruction}, 
            {"type" : "image", "image" : load_image_from_dict(sample["image"])} ] 
        }, 
        { "role" : "assistant", 
          "content" : [ 
            {"type" : "text",  "text"  : sample["caption"]} ] 
        }, 
    ]  
    return { "messages" : conversation } 
pass 
converted_dataset = [convert_to_conversation(sample) for sample in dataset] 
converted_dataset[0] 
from PIL import Image 
import io 
 
def bytes_to_image(image_bytes: bytes) -> Image.Image: 
    return Image.open(io.BytesIO(image_bytes)) 
FastVisionModel.for_inference(model) # Enable for inference! 
 
image = bytes_to_image(dataset[0]["image"]['bytes']) 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
messages = [ 
    {"role": "user", "content": [ 
        {"type": "image"}, 
        {"type": "text", "text": instruction} 
    ]} 
] 
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True) 
inputs = tokenizer( 
    image, 
    input_text, 
    add_special_tokens = False, 
    return_tensors = "pt", 
).to("cuda") 
 
from transformers import TextStreamer 
text_streamer = TextStreamer(tokenizer, skip_prompt = True) 
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128, 
                   use_cache = True, temperature = 1.5, min_p = 0.1) 
from unsloth import is_bf16_supported 
from unsloth.trainer import UnslothVisionDataCollator 
from trl import SFTTrainer, SFTConfig 
 
FastVisionModel.for_training(model) # Enable for training! 
 
trainer = SFTTrainer( 
    model = model, 
    tokenizer = tokenizer, 
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use! 
    train_dataset = converted_dataset, 
    args = SFTConfig( 
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4, 
        warmup_steps = 5, 
        max_steps = 50, 
        # num_train_epochs = 1,  
        learning_rate = 2e-4, 
        fp16 = not is_bf16_supported(), 
        bf16 = is_bf16_supported(), 
        logging_steps = 1, 
        optim = "adamw_8bit", 
        weight_decay = 0.01, 
        lr_scheduler_type = "linear", 
        seed = 3407, 
        output_dir = "outputs", 
        report_to = "none",      
 
        remove_unused_columns = False, 
        dataset_text_field = "", 
        dataset_kwargs = {"skip_prepare_dataset": True}, 
        dataset_num_proc = 4, 
        max_seq_length = 2048, 
    ), 
) 
trainer_stats = trainer.train() 
 
import matplotlib.pyplot as plt 
 
# Extract loss values from log history 
logs = trainer.state.log_history 
steps = [] 
losses = [] 
  
for log in logs: 
    if 'loss' in log: 
        steps.append(log['step']) 
        losses.append(log['loss']) 
 
# Plot the graph 
plt.figure(figsize=(8, 5)) 
plt.plot(steps, losses, label='Training Loss', color='blue', marker='o') 
plt.xlabel("Training Step") 
plt.ylabel("Loss") 
plt.title("Training Loss over Steps") 
plt.grid(True) 
plt.legend() 
plt.tight_layout() 
plt.show() 
FastVisionModel.for_inference(model)  
image = bytes_to_image(dataset[10]["image"]['bytes']) 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
messages = [ 
    {"role": "user", "content": [ 
        {"type": "image"}, 
        {"type": "text", "text": instruction} 
    ]} 
] 
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True) 
inputs = tokenizer( 
    image, 
    input_text, 
    add_special_tokens = False, 
    return_tensors = "pt", 
).to("cuda") 
 
from transformers import TextStreamer 
text_streamer = TextStreamer(tokenizer, skip_prompt = True) 
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128, 
                   use_cache = True, temperature = 1.5, min_p = 0.1) 
   
model.push_to_hub("santhoshk3688/generate-xray-report", token = 
"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")  
processor.push_to_hub("santhoshk3688/generate-xray-report", token = 
"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")!pip install pip3-autoremove 
!pip-autoremove torch torchvision torchaudio -y 
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 
!pip install unsloth 
from unsloth import FastVisionModel 
import torch 
 
model, tokenizer = FastVisionModel.from_pretrained( 
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", 
    load_in_4bit = True, 
    use_gradient_checkpointing = "unsloth", 
) 
 
model = FastVisionModel.get_peft_model( 
    model, 
    finetune_vision_layers     = False, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 
 
    r = 16, 
    lora_alpha = 16, 
    lora_dropout = 0, 
    bias = "none", 
    random_state = 3407, 
    use_rslora = False, 
    loftq_config = None, 
) 
from datasets import load_dataset 
dataset = load_dataset("Santhosh1705kumar/radiology-reports-chest", split = "train") 
dataset 
from PIL import Image 
from io import BytesIO 
 
def load_image_from_dict(image_dict): 
    """ 
    Converts a dictionary containing image bytes into a PIL Image. 
 
    Args: 
        image_dict (dict): A dictionary with a 'bytes' key containing image data. 
 
    Returns: 
        PIL.Image.Image: The decoded image. 
    """ 
    image_bytes = image_dict["bytes"] 
    return Image.open(BytesIO(image_bytes)) 
for i in range(100, 120): 
  image = load_image_from_dict(dataset[i]["image"]) 
  display(image) 
dataset[0]["caption"] 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
def convert_to_conversation(sample): 
    conversation = [ 
        { "role": "user", 
          "content" : [ 
            {"type" : "text",  "text"  : instruction}, 
            {"type" : "image", "image" : load_image_from_dict(sample["image"])} ] 
        }, 
        { "role" : "assistant", 
          "content" : [ 
            {"type" : "text",  "text"  : sample["caption"]} ] 
        }, 
    ] 
    return { "messages" : conversation } 
pass 
converted_dataset = [convert_to_conversation(sample) for sample in dataset] 
converted_dataset[0] 
from PIL import Image 
import io 
 
def bytes_to_image(image_bytes: bytes) -> Image.Image: 
    return Image.open(io.BytesIO(image_bytes)) 
FastVisionModel.for_inference(model) # Enable for inference! 
 
image = bytes_to_image(dataset[0]["image"]['bytes']) 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
messages = [ 
    {"role": "user", "content": [ 
        {"type": "image"}, 
        {"type": "text", "text": instruction} 
    ]} 
] 
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True) 
inputs = tokenizer( 
    image, 
    input_text, 
    add_special_tokens = False, 
    return_tensors = "pt", 
).to("cuda") 
 
from transformers import TextStreamer 
text_streamer = TextStreamer(tokenizer, skip_prompt = True) 
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128, 
                   use_cache = True, temperature = 1.5, min_p = 0.1) 
from unsloth import is_bf16_supported 
from unsloth.trainer import UnslothVisionDataCollator 
from trl import SFTTrainer, SFTConfig 
 
FastVisionModel.for_training(model) # Enable for training! 
 
trainer = SFTTrainer( 
    model = model, 
    tokenizer = tokenizer, 
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use! 
    train_dataset = converted_dataset, 
    args = SFTConfig( 
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4, 
        warmup_steps = 5, 
        max_steps = 50, 
        # num_train_epochs = 1,  
        learning_rate = 2e-4, 
        fp16 = not is_bf16_supported(), 
        bf16 = is_bf16_supported(), 
        logging_steps = 1, 
        optim = "adamw_8bit", 
        weight_decay = 0.01, 
        lr_scheduler_type = "linear", 
        seed = 3407, 
        output_dir = "outputs", 
        report_to = "none",      
 
        remove_unused_columns = False, 
        dataset_text_field = "", 
        dataset_kwargs = {"skip_prepare_dataset": True}, 
        dataset_num_proc = 4, 
        max_seq_length = 2048, 
    ), 
) 
trainer_stats = trainer.train() 
 
import matplotlib.pyplot as plt 
 
# Extract loss values from log history 
logs = trainer.state.log_history 
steps = [] 
losses = [] 
 
for log in logs: 
    if 'loss' in log: 
        steps.append(log['step']) 
        losses.append(log['loss']) 
 
# Plot the graph 
plt.figure(figsize=(8, 5)) 
plt.plot(steps, losses, label='Training Loss', color='blue', marker='o') 
plt.xlabel("Training Step") 
plt.ylabel("Loss") 
52  
plt.title("Training Loss over Steps") 
plt.grid(True) 
plt.legend() 
plt.tight_layout() 
plt.show() 
FastVisionModel.for_inference(model)  
image = bytes_to_image(dataset[10]["image"]['bytes']) 
instruction = "You are an expert radiographer. Describe accurately what you see in this image." 
 
messages = [ 
    {"role": "user", "content": [ 
        {"type": "image"}, 
        {"type": "text", "text": instruction} 
    ]} 
] 
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True) 
inputs = tokenizer( 
    image, 
    input_text, 
    add_special_tokens = False, 
    return_tensors = "pt", 
).to("cuda") 
 
from transformers import TextStreamer 
text_streamer = TextStreamer(tokenizer, skip_prompt = True) 
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128, 
                   use_cache = True, temperature = 1.5, min_p = 0.1) 
 
model.push_to_hub("santhoshk3688/generate-xray-report", token = 
"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")  
processor.push_to_hub("santhoshk3688/generate-xray-report", token = 
"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")