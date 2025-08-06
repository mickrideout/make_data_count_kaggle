import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. Define the base model and the adapter path
base_model_name = "Universal-NER/UniNER-7B-type-sup"
adapter_path = "/home/mick/tmp/kaggle/universal-ner/src/train/saved_models/universalner"
merged_model_path = "/home/mick/tmp/merged-model"

# 2. Load the base model
# Use AutoModelForCausalLM instead of AutoModelForTokenClassification
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Load the PEFT model (merges the adapter layers on top)
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. Merge the adapter weights into the base model
# This operation is done in-place
merged_model = peft_model.merge_and_unload()

# 5. Save the merged model
# This saves the full model, which can now be loaded without PEFT
merged_model.save_pretrained(merged_model_path)

# You might also want to save the tokenizer for easy use later
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved to {merged_model_path}")