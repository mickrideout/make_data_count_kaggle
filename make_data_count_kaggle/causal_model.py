import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset as HFDataset
import torch
from pydantic import BaseModel, Field
from typing import List
import re

prompt_text = """

You are an expert in extracting and categorizing dataset mentions from research papers and policy documents. Your task is to **identify and extract all valid dataset mentions** and ensure their type is correctly classified. You to read, parse and then perform the extraction of the text in the ## **Text** section below. Provide your reponse in json format defined in the section ### **Extraction Schema**


### **What Qualifies as a Dataset?**
A dataset is a structured collection of data used for empirical research, analysis, or policy-making. Examples include:
- **Surveys & Census Data** (e.g., LSMS, DHS, national census records)
- **Indicators & Indexes** (e.g., HDI, GFSI, WDI, ND-GAIN, EPI)
- **Geospatial & Environmental Data** (e.g., OpenStreetMap, Sentinel-2 imagery)
- **Economic & Trade Data** (e.g., UN Comtrade, Balance of Payments Statistics)
- **Health & Public Safety Data** (e.g., epidemiological surveillance, crime reports)
- **Time-Series & Energy Data** (e.g., climate projections, electricity demand records)
- **Transport & Mobility Data** (e.g., road accident statistics, smart city traffic flow)
- **Other emerging dataset types** as identified in the text.
- **DOI Reference** is used to reference both datasets and papers. They are of the form  https://doi.org/[prefix]/[suffix] an example being https://doi.org/10.1371/journal.pone.0303785.

**Important:**  
If the dataset does not fit into the examples above, infer the **most appropriate category** from the context and **create a new `"data_type"` if necessary**.

### **What Should NOT Be Extracted?**
- Do **not** extract mentions that do not explicitly refer to a dataset by name, or a dataset by DOI reference.
- Do **not** extract DOI metions that refer to a paper and not a dataset. Use context to infer the DOI reference type.
- Do **not** extract mentions that do not clearly refer to a dataset, including, but not limited to:
1. **Organizations & Institutions** (e.g., WHO, IMF, UNDP, "World Bank data" unless it explicitly refers to a dataset)
2. **Reports & Policy Documents** (e.g., "Fiscal Monitor by the IMF", "IEA Energy Report"; only extract if the dataset itself is referenced)
3. **Generic Mentions of Data** (e.g., "various sources", "survey results from multiple institutions")
4. **Economic Models & Policy Frameworks** (e.g., "GDP growth projections", "macroeconomic forecasts")
5. **Legislation & Agreements** (e.g., "Paris Agreement", "General Data Protection Regulation")

### **Rules for Extraction**
1. **Classify `"type"` Correctly**
   - `"Primary"`: The dataset is used for direct analysis in the document. It is raw or processed data generated as part of this paper, specifically for this study
   - `"Secondary"`: The dataset is referenced to validate or compare findings. It is  raw or processed data derived or reused from existing records or published data sources.

   **Examples:**
   - `"The LSMS-ISA data is analyzed to assess the impact of agricultural practices on productivity."` → `"Primary"`
   -  "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9." → `"Primary"`
   - `"Our results align with previous studies that used LSMS-ISA."` → `"Secondary"`
   - "“The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.”` → `"Secondary"`.

### **Extraction Schema**
Each extracted dataset should have the following fields:
- `dataset_name`: Exact dataset name from the text (**no paraphrasing**).
- `dataset_type`: **Primary / Secondary**

### **Example Response**
- If no datasets are found, return an empty json list `[]`.
- An example response is:
[
    {
        "dataset_name": "LSMS-ISA",
        "dataset_type": "Primary"
    },
    {
        "dataset_name": "https://doi.org/10.1371/journal.pone.0303785",
        "dataset_type": "Secondary"
    }
]


## Text
{article_text}


"""

class Dataset(BaseModel):
    dataset_name: str = Field(description="The name of the dataset")
    dataset_type: str = Field(description="The type of the dataset")

class Response(BaseModel):
    datasets: List[Dataset] = Field(description="The list of datasets")


def load_article_text(article_id, output_dir):
    """
    Load article text from {output_dir}/{article_id}.txt
    
    Args:
        article_id (str): The article identifier
        output_dir (str): Directory containing article text files
        
    Returns:
        str: Article text content
    """
    text_file = Path(output_dir) / f"{article_id}.txt"
    if not text_file.exists():
        raise FileNotFoundError(f"Article text file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_training_dataset(dataset_dict, output_dir):
    """
    Create training dataset with article_id and expected datasets for on-the-fly prompt generation.
    
    Args:
        dataset_dict: HuggingFace DatasetDict containing train/test splits
        output_dir (str): Directory containing article text files
        
    Returns:
        HFDataset: HuggingFace dataset with article_id and completion data
    """
    training_data = []
    
    # Group by article_id to create complete responses
    article_datasets = {}
    for item in dataset_dict['train']:
        article_id = item['article_id']
        if article_id not in article_datasets:
            article_datasets[article_id] = []
        article_datasets[article_id].append({
            "dataset_name": item['dataset_id'],
            "dataset_type": item['type']
        })
    
    # Create dataset with just article_id and expected output
    for article_id, datasets in article_datasets.items():
        # Check if article text exists
        text_file = Path(output_dir) / f"{article_id}.txt"
        if text_file.exists():
            training_data.append({
                'article_id': article_id,
                'completion': json.dumps(datasets, indent=2)
            })
        else:
            print(f"Warning: Article text not found for {article_id}, skipping")
    
    return HFDataset.from_list(training_data)


def train_causal_model(dataset_dict, output_dir, model_output_dir):
    """
    Train a causal language model using microsoft/Phi-4-mini-instruct.
    
    Args:
        dataset_dict: HuggingFace DatasetDict containing train/test splits
        output_dir (str): Directory containing article text files and prompt template
        model_output_dir (str): Directory to save the trained model
    """
    # Create model output directory
    model_output_path = Path(model_output_dir)
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer with larger context window
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Large context window, instruct tuned
    print(f"Loading model and tokenizer: {model_name}")
    
    # Use 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',  # Auto device mapping
        trust_remote_code=True
    )
    
    # Add LoRA adapters for memory-efficient fine-tuning
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # Enable gradients for LoRA parameters
    model.enable_input_require_grads()
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create training dataset
    training_dataset = create_training_dataset(dataset_dict, output_dir)
    
    print(f"Created training dataset with {len(training_dataset)} examples")
    
    # Define formatting function that creates prompts on-the-fly
    def formatting_func(example):
        try:
            # Load article text on-the-fly
            article_text = load_article_text(example['article_id'], output_dir)
            # Create prompt by substituting article text
            prompt = prompt_text.replace('{article_text}', article_text)
            # Combine prompt and completion
            return f"{prompt}{example['completion']}"
        except FileNotFoundError:
            # Fallback if article text is missing
            return f"{prompt_text.replace('{article_text}', '[Article text not found]')}{example['completion']}"
    
    # Training arguments optimized for LoRA + quantization
    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        num_train_epochs=3,  # More epochs since LoRA needs more training
        per_device_train_batch_size=2,  # Can increase with quantization
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2*4 = 8
        warmup_steps=20,
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,  # Keep only 1 checkpoint
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Enable to save memory
        bf16=True,  # Use bf16
        max_grad_norm=1.0,  # Add gradient clipping
        report_to=None,  # Disable wandb logging
        dataloader_num_workers=0,  # Disable multiprocessing
        optim="paged_adamw_32bit",  # Memory-efficient optimizer
        learning_rate=2e-4,  # Higher learning rate for LoRA
    )
    
    # Create SFTTrainer with basic parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        formatting_func=formatting_func,
    )
    
    # Train model
    print("Starting model training...")
    trainer.train()
    
    # Save final model
    print(f"Saving trained model to {model_output_path}")
    trainer.save_model()
    tokenizer.save_pretrained(str(model_output_path))
    
    print("Model training completed successfully!")
    
    return str(model_output_path)


def run_inference(dataset_dict, output_dir, model_dir):
    """
    Run inference on test dataset using the trained causal model.
    
    Args:
        dataset_dict: HuggingFace DatasetDict containing train/test splits
        output_dir (str): Directory containing article text files
        model_dir (str): Directory containing the trained model
        
    Returns:
        List[Response]: List of Response objects with extracted datasets for each article
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Load trained model and tokenizer with same quantization as training
    print(f"Loading trained model from {model_path}")
    
    # Use 4-bit quantization to reduce memory usage (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',  # Enable automatic device mapping
        low_cpu_mem_usage=True
    )
    
    # Device will be set automatically by device_map='auto'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get unique article IDs from test dataset
    test_article_ids = list(set(item['article_id'] for item in dataset_dict['test']))
    
    results = []
    
    for article_id in test_article_ids:
        try:
            # Load article text
            article_text = load_article_text(article_id, output_dir)
            
            # Create prompt
            prompt = prompt_text.replace('{article_text}', article_text)
            
            # Create prompt without truncation
            prompt = prompt_text.replace('{article_text}', article_text)
            
            # Tokenize with limits appropriate for 16384 token limit
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                padding=False  # No padding to avoid issues
            )
            
            # Move inputs to GPU device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Create attention mask manually
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Check if input is too long (should not happen due to truncation)
                # Generate response with very conservative settings
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=512,  # More room for JSON response
                        do_sample=False,    # Greedy decoding
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    
                    # Decode generated text safely
                    # Check the shape of outputs tensor
                    if outputs.dim() > 1 and outputs.shape[0] > 0:
                        # outputs is [batch_size, sequence_length], take first batch
                        generated_tokens = outputs[0]
                    else:
                        # outputs is already 1D or empty
                        generated_tokens = outputs
                    
                    # Get input length to extract only new tokens
                    input_length = inputs['input_ids'].shape[1]
                    
                    # Extract only the newly generated tokens
                    if generated_tokens.shape[0] > input_length:
                        new_tokens = generated_tokens[input_length:]
                        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    else:
                        response_text = ""
                    
                    # Parse JSON response
                    datasets = parse_model_response(response_text)
                    
                except Exception as gen_error:
                    print(f"Generation error for {article_id}: {gen_error}")
                    import traceback
                    traceback.print_exc()
                    print(f"Input shape: {inputs['input_ids'].shape}")
                    print(f"Input length: {inputs['input_ids'].shape[1]}")
                    datasets = []
            
            # Create Response object with pydantic validation
            response = Response(datasets=datasets)
            results.append({
                'article_id': article_id,
                'response': response
            })
            
            print(f"Processed article {article_id}: found {len(datasets)} datasets")
            
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            # Create empty response for failed articles
            response = Response(datasets=[])
            results.append({
                'article_id': article_id,
                'response': response
            })
    
    return results


def parse_model_response(response_text):
    """
    Parse the model's JSON response and extract datasets.
    Handles Deepseek R1's <think> tags by extracting only the JSON portion.
    
    Args:
        response_text (str): Generated response from the model
        
    Returns:
        List[Dataset]: List of Dataset objects
    """
    try:
        # Remove <think> tags and their content first
        clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Also handle unclosed think tags
        clean_text = re.sub(r'<think>.*$', '', clean_text, flags=re.DOTALL)
        
        # Try to find JSON array in the cleaned response
        json_match = re.search(r'\[.*?\]', clean_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            datasets_data = json.loads(json_str)
            
            # Convert to Dataset objects
            datasets = []
            for item in datasets_data:
                if isinstance(item, dict) and 'dataset_name' in item and 'dataset_type' in item:
                    dataset = Dataset(
                        dataset_name=item['dataset_name'],
                        dataset_type=item['dataset_type']
                    )
                    datasets.append(dataset)
            
            return datasets
        else:
            print(f"No valid JSON found in response after cleaning: {clean_text[:200]}...")
            return []
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Cleaned response text: {clean_text[:200] if 'clean_text' in locals() else response_text[:200]}...")
        return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []