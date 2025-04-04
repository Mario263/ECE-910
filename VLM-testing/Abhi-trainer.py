import json
from transformers import LlavaProcessor, LlavaForConditionalGeneration,TrainingArguments,EarlyStoppingCallback
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from trl import SFTTrainer
import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else cpu
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,load_in_4bit=True)
model.to(device)


def read_json_file(filename="limited_processed_data.json"):
    """Read a JSON file and return its content as a Python object (list/dict)."""
    with open(filename, 'r') as f:
        data = json.load(f)
        for entry in data:
            yield entry

def collate_fn(batch):
    batch_texts = []
    batch_images = []

        
    for entry in batch:
        batch_texts.append(entry["text"])
        for image in entry["frames"]:
            batch_images.append(Image.open(image))

    # Process batch using the processor
    processed_data = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )
    labels = processed_data["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    processed_data["labels"] = labels
    return processed_data


def create_hugging_face_dataset(generator_func):
    return Dataset.from_generator(generator_func)

def add_text(data):
    data["text"] = data["prompt"]
    return data


if __name__=="__main__":
    dataset = load_dataset("json", data_files="limited_processed_data.json", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(add_text)
    train_test_split = dataset.train_test_split(test_size=0.2)
    target_modules = ["q_proj","v_proj","fc1","fc2"]
    peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, peft_config).to(device)
    print("Trainable Parameters are as follows:")
    peft_model.print_trainable_parameters()
    print("Setting up training Arguments")
    training_args = TrainingArguments(
    output_dir='./abhi-2',
    num_train_epochs=6,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=10,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=5, 
    logging_steps=1,
    logging_strategy="steps",
    gradient_checkpointing=True,
    metric_for_best_model="loss", 
    load_best_model_at_end=True,
    save_steps=500)
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)
    print("Starting Training")
    trainer.train()
    
    
