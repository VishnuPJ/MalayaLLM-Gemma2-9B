import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments

wandb.login(key="<your wandb key>")
wandb.init(project="Gemma_2_9B_MalayaLLM", name = "Pretrain")

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "unsloth/gemma-2-9b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", 
)

print("length of tokenizer before",len(tokenizer))
print(tokenizer.tokenize("നിങ്ങൾ ചിന്തിക്കുന്നത് ഞങ്ങളെ അറിയിക്കുക"))

#Uncomment the below lines , if you want to add additional sentencepiece tokens
"""
# all_vocabs = ["malayalam_bpe_spe.model"]

#Add Malayalam tokens
##This should be done before calling FastLanguageModel.get_peft_model()
# mlm_sp_model = spm.SentencePieceProcessor()
# for v in all_vocabs:
#     mlm_sp_model.Load(v)
#     vocab = [str(mlm_sp_model.decode(i)) for i in range(len(mlm_sp_model))]
#     tokenizer.add_tokens(vocab)
# model.resize_token_embeddings(len(tokenizer))
"""

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

wikipedia_prompt = """തലക്കെട്ട്:{}

ലേഖനം:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = examples["title"]
    texts  = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
pass

dataset = load_dataset("wikimedia/wikipedia", "20231101.ml", split = "train",)
#dataset = load_dataset("VishnuPJ/Malayalam_CultureX_IndicCorp_SMC", split = "train")

print("length of tokenizer after",len(tokenizer))
print(tokenizer.tokenize("നിങ്ങൾ ചിന്തിക്കുന്നത് ഞങ്ങളെ അറിയിക്കുക"))

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 24,
    # formatting_func = formatting_func,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        # max_steps = 240,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs_pretrain",
	    report_to="wandb"
    ),
)

#@title Show current memory stats
#gpu_stats = torch.cuda.get_device_properties(0)
#start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
#max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
#print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
#print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
# trainer_stats = trainer.train("/home/ubuntu/gemma2/checkpoint-10723")
wandb.finish()
model.save_pretrained_merged("Merged_Gemma_2_9B_MalayaLLM_Pretrain", tokenizer, save_method = "merged_16bit",)

