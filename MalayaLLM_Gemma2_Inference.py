import torch
import sentencepiece as spm
from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "Merged_Gemma_2_9B_MalayaLLM_Finetune", #"VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",

print(tokenizer.tokenize("നിങ്ങൾ ചിന്തിക്കുന്നത് ഞങ്ങളെ അറിയിക്കുക"))

alpaca_prompt_1 = """ഒരു  ചുമതല  വിവരിക്കുന്ന  ഒരു  നിർദ്ദേശം  ചുവടെയുണ്ട്.
 അഭ്യർത്ഥന  ശരിയായി  പൂർത്തിയാക്കുന്ന  ഒരു  പ്രതികരണം  എഴുതുക.".

### നിർദ്ദേശം:
{}

### ഇൻപുട്ട്:
{}

### പ്രതികരണം:
{}"""

alpaca_prompt_2 = """ഒരു  ചുമതല  വിവരിക്കുന്ന  ഒരു  നിർദ്ദേശം  ചുവടെയുണ്ട്.
 അഭ്യർത്ഥന  ശരിയായി  പൂർത്തിയാക്കുന്ന  ഒരു  പ്രതികരണം  എഴുതുക.".

### നിർദ്ദേശം:
{}

### പ്രതികരണം:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# inputs = tokenizer([
#     alpaca_prompt_1.format(
#         """ഈ പറയുന്നതിൽ സൂര്യൻ ഉദിക്കുന്ന ദിശ ഏതെന്നു പറയുക.""", # instruction
#         """ വടക്ക് , കിഴക്ക് , തെക്ക് ,പടിഞ്ഞാറു , താഴെ """, # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")

inputs = tokenizer([
    alpaca_prompt_2.format(
        """സൂര്യൻ ഉദിക്കുന്ന ദിശ ഏതെന്നു പറയുക.""",
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")


outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs))

