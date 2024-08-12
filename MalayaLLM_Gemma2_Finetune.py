import torch
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from unsloth import UnslothTrainer, UnslothTrainingArguments
import wandb

wandb.login(key="<your wandb key>")
wandb.init(project="Gemma_2_9B_MalayaLLM", name = "Finetune")

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "Merged_Gemma_2_9B_MalayaLLM_Pretrain",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

aya_dataset = load_dataset("CohereForAI/aya_dataset", split = "train")

all_codes = aya_dataset.to_pandas().groupby(["language", "language_code"]).agg({"language" : "count"})
all_codes.columns = ["Count"]
all_codes = all_codes.sort_values("Count", ascending = False)

aya_dataset = aya_dataset.filter(lambda x: x["language_code"] == "mal")

alpaca_prompt = """ഒരു  ചുമതല  വിവരിക്കുന്ന  ഒരു  നിർദ്ദേശം  ചുവടെയുണ്ട്.
 അഭ്യർത്ഥന  ശരിയായി  പൂർത്തിയാക്കുന്ന  ഒരു  പ്രതികരണം  എഴുതുക.".

### നിർദ്ദേശം:
{}

### ഇൻപുട്ട്:
{}

### പ്രതികരണം:
{}"""

response_template = " ### പ്രതികരണം:"

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["inputs"]
    outputs      = examples["targets"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # Careful Aya Dataset does not have an input!
        text = alpaca_prompt.format(instruction, "", output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

aya_dataset = aya_dataset.map(formatting_prompts_func, batched = True,)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = aya_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 24,
    #data_collator=collator,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use num_train_epochs and warmup_ratio for longer runs!
        # max_steps = 120,
        warmup_steps = 100,
        # warmup_ratio = 0.1,
        num_train_epochs = 10,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 2e-4,
        embedding_learning_rate = 2e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_finetune",
        report_to="wandb"
    ),
)


trainer_stats = trainer.train()
model.save_pretrained_merged("Merged_Gemma_2_9B_MalayaLLM_Finetune", tokenizer, save_method = "merged_16bit",)
