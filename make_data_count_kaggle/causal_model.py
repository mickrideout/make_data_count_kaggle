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

MAX_NEW_TOKENS = 4000
MAX_TOKEN_LENGTH = 100000
MAX_CHARS_PER_ARTICLE = 100000

# Try to import Outlines, but handle gracefully if not available
try:
    from outlines import models, Generator
    OUTLINES_AVAILABLE = True
except ImportError:
    print("Warning: Outlines library not available. Structured generation will be disabled.")
    OUTLINES_AVAILABLE = False

# Global cache for Outlines model to prevent recreating it
_OUTLINES_MODEL_CACHE = {}
_OUTLINES_GENERATOR_CACHE = {}

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
- If the dataset does not fit into the examples above, infer the **most appropriate category** from the context and **create a new `"data_type"` if necessary**.
- The final line of your response **MUST** be a json array of the datasets.

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
{
    "datasets": 
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
}


## Text
{article_text}

## Json Response:

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
        response = {
            "datasets": article_datasets[article_id]
        }
    
    # Create dataset with just article_id and expected output
    for article_id, datasets in article_datasets.items():
        # Check if article text exists
        text_file = Path(output_dir) / f"{article_id}.txt"
        if text_file.exists():
            training_data.append({
                'article_id': article_id,
                'completion': json.dumps(response, indent=2)
            })
        else:
            print(f"Warning: Article text not found for {article_id}, skipping")
    
    return HFDataset.from_list(training_data)


def train_causal_model(dataset_dict, output_dir, model_dir):
    """
    Train a causal language model
    
    Args:
        dataset_dict: HuggingFace DatasetDict containing train/test splits
        output_dir (str): Directory containing article text files and prompt template
        model_dir (str): Directory to save the trained model
    """

    
    # Load model and tokenizer with larger context window
  # Large context window, instruct tuned
    print(f"Loading model and tokenizer: {model_dir}")
    
    # Use 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map='auto',  # Auto device mapping
        trust_remote_code=True,
        local_files_only=True
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
    
    # Define formatting function that creates prompts on-the-fly using chat template
    def formatting_func(example):
        try:
            # Load article text on-the-fly
            article_text = load_article_text(example['article_id'], output_dir)
            # Create prompt by substituting article text
            user_prompt = prompt_text.replace('{article_text}', article_text)
            
            # Format as chat messages
            messages = [
                {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": example['completion']}
            ]
            
            # Apply chat template
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except FileNotFoundError:
            # Fallback if article text is missing
            fallback_prompt = prompt_text.replace('{article_text}', '[Article text not found]')
            messages = [
                {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
                {"role": "user", "content": fallback_prompt},
                {"role": "assistant", "content": example['completion']}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
    
    # Training arguments optimized for LoRA + quantization
    training_args = TrainingArguments(
        output_dir=str(model_dir),
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
    
    
    # First save the LoRA adapters
    adapter_path = Path(f"{model_dir}/adapters")
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_path))
    
    # Merge LoRA adapters with base model for standalone inference
    print("Merging LoRA adapters with base model for offline compatibility...")
    try:
        merged_model = model.merge_and_unload()
        # Save the merged model
        merged_model.save_pretrained(str(model_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(model_dir))
        print("Successfully saved merged model for offline use")
    except Exception as e:
        print(f"Warning: Failed to merge model ({e}). Saving adapters only.")
        # Fallback: save adapters and base model config
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        
        # Save base model name for reference
        config_file = f"{model_dir}/base_model.txt"
        with open(config_file, 'w') as f:
            f.write(model_dir)
    
    print("Model training completed successfully!")
    
    return str(model_dir)


def test_outlines_integration(model, tokenizer):
    """
    Test Outlines integration with a simple prompt to verify it's working correctly.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        
    Returns:
        bool: True if Outlines integration works, False otherwise
    """
    if not OUTLINES_AVAILABLE:
        print("Outlines not available, skipping integration test")
        return False
    
    try:
        print("Testing Outlines integration...")
        outlines_model = models.from_transformers(model, tokenizer)
        json_generator = Generator(outlines_model, Response)
        
        # Simple test prompt using chat template
        user_prompt = prompt_text.replace('{article_text}', "This paper uses LSMS-ISA data for analysis.")
        messages = [
            {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
            {"role": "user", "content": user_prompt}
        ]
        
        test_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = json_generator(test_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.9, repetition_penalty=1.1)
        print(f"Test response type: {type(response)}")
        print(f"Test response: {response}")
        
        # Validate response format
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                print(f"Successfully parsed string response as JSON: {parsed}")
                return True
            except json.JSONDecodeError:
                print(f"Failed to parse string response as JSON: {response}")
                return False
        elif hasattr(response, 'datasets') or isinstance(response, (dict, list)):
            print(f"Response has expected format: {type(response)}")
            return True
        else:
            print(f"Unexpected response format: {type(response)}")
            return False
            
    except Exception as e:
        print(f"Outlines integration test failed: {e}")
        return False


def parse_model_response(response_text):
    """
    Parse model response text to extract datasets.
    
    Args:
        response_text (str): The generated response text
        
    Returns:
        List[Dataset]: List of extracted datasets
    """
    datasets = []
    try:
        # Look for JSON array in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    if isinstance(item, dict) and 'dataset_name' in item and 'dataset_type' in item:
                        datasets.append(Dataset(
                            dataset_name=item['dataset_name'],
                            dataset_type=item['dataset_type']
                        ))
                print(f"Extracted {len(datasets)} datasets from response")
            else:
                print(f"Parsed data is not a list: {type(parsed_data)}")
        else:
            print(f"No JSON array found in response")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from response: {e}")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return datasets


def debug_outlines_response(model, tokenizer, test_text):
    """
    Debug Outlines response generation with detailed logging.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        test_text (str): Test text to use
        
    Returns:microsoft/Phi-4-mini-instruct
    """
    if not OUTLINES_AVAILABLE:
        return {"error": "Outlines not available"}
    
    debug_info = {}
    
    try:
        print("Creating Outlines model...")
        outlines_model = models.from_transformers(model, tokenizer)
        print("✓ Outlines model created")
        
        print("Creating JSON generator...")
        json_generator = Generator(outlines_model, Response)
        print("✓ JSON generator created")
        
        # Create test prompt using chat template
        user_prompt = prompt_text.replace('{article_text}', test_text)
        messages = [
            {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
            {"role": "user", "content": user_prompt}
        ]
        
        test_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Test prompt length: {len(test_prompt)} characters")
        
        print("Generating response...")
        response = json_generator(test_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.9, repetition_penalty=1.1)
        
        debug_info["response_type"] = type(response).__name__
        response_str = str(response)
        debug_info["response_str"] = response_str[:500]
        debug_info["response_length"] = len(response_str)
        
        response_str = str(response)
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response_str)} characters")
        print(f"Response preview: {response_str[:200]}{'...' if len(response_str) > 200 else ''}")
        
        # Try different parsing approaches
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                debug_info["parsed_json"] = parsed
                print(f"✓ Successfully parsed as JSON: {parsed}")
            except json.JSONDecodeError as e:
                debug_info["json_error"] = str(e)
                print(f"✗ Failed to parse as JSON: {e}")
        
        return debug_info
        
    except Exception as e:
        debug_info["error"] = str(e)
        debug_info["error_type"] = type(e).__name__
        print(f"✗ Error in debug_outlines_response: {e}")
        return debug_info


def get_cached_outlines_generator(model, tokenizer, model_id):
    """
    Get a cached Outlines generator to avoid recreating models.
    
    Args:
        model: The base model
        tokenizer: The tokenizer
        model_id: Unique identifier for this model instance
        
    Returns:
        Generator: Cached or newly created Outlines generator
    """
    global _OUTLINES_MODEL_CACHE, _OUTLINES_GENERATOR_CACHE
    
    if not OUTLINES_AVAILABLE:
        return None
        
    cache_key = f"{model_id}_{id(model)}"
    
    if cache_key not in _OUTLINES_GENERATOR_CACHE:
        try:
            # Check memory before creating
            if torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info()
                available_memory = memory_info[0] / 1024**3
                if available_memory < 8.0:
                    print(f"Insufficient memory ({available_memory:.2f} GB) for Outlines caching")
                    return None
            
            print(f"Creating new Outlines generator for cache key: {cache_key}")
            outlines_model = models.from_transformers(model, tokenizer)
            json_generator = Generator(outlines_model, Response)
            
            _OUTLINES_MODEL_CACHE[cache_key] = outlines_model
            _OUTLINES_GENERATOR_CACHE[cache_key] = json_generator
            
        except Exception as e:
            print(f"Failed to create cached Outlines generator: {e}")
            return None
    
    return _OUTLINES_GENERATOR_CACHE.get(cache_key)


def clear_outlines_cache():
    """
    Clear the Outlines model cache to free memory.
    """
    global _OUTLINES_MODEL_CACHE, _OUTLINES_GENERATOR_CACHE
    
    # Clean up cached objects
    for key in list(_OUTLINES_GENERATOR_CACHE.keys()):
        try:
            del _OUTLINES_GENERATOR_CACHE[key]
        except:
            pass
    
    for key in list(_OUTLINES_MODEL_CACHE.keys()):
        try:
            del _OUTLINES_MODEL_CACHE[key]
        except:
            pass
    
    _OUTLINES_MODEL_CACHE.clear()
    _OUTLINES_GENERATOR_CACHE.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    print("Outlines cache cleared")


def check_memory_safety(required_gb=8.0, operation="operation"):
    """
    Check if there's sufficient GPU memory for the operation.
    
    Args:
        required_gb (float): Required memory in GB
        operation (str): Description of the operation
        
    Returns:
        bool: True if sufficient memory, False otherwise
    """
    if torch.cuda.is_available():
        memory_info = torch.cuda.mem_get_info()
        available_memory = memory_info[0] / 1024**3
        total_memory = memory_info[1] / 1024**3
        used_memory = total_memory - available_memory
        
        print(f"Memory check for {operation}: {used_memory:.2f}/{total_memory:.2f} GB used, {available_memory:.2f} GB available")
        
        if available_memory < required_gb:
            print(f"WARNING: Insufficient memory for {operation} (need {required_gb:.1f} GB, have {available_memory:.2f} GB)")
            return False
        return True
    return True  # Assume OK if no GPU


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
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory cleared. Total memory: {total_memory:.2f} GB")
        check_memory_safety(required_gb=4.0, operation="model loading")
    
    # Load trained model and tokenizer with same quantization as training
    print(f"Loading trained model from {model_path}")
    
    # Use 4-bit quantization to reduce memory usage (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    
    # Ensure tokenizer has proper configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with error handling - try merged model first, then adapters
    try:
        # Try loading merged model first (preferred for Kaggle offline)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        )
        print("Successfully loaded merged model")
    except Exception as e:
        print(f"Failed to load merged model: {e}")
        # Fallback: try loading base model + adapters
        base_model_file = Path(model_dir) / "base_model.txt"
        if base_model_file.exists():
            with open(base_model_file, 'r') as f:
                base_model_name = f.read().strip()
            print(f"Attempting to load base model + adapters: {base_model_name}")
            try:
                # This would require internet access in Kaggle - not ideal
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    trust_remote_code=True,
                    local_files_only=True
                )
                # Load adapters
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_dir,local_files_only=True, trust_remote_code=True)
                print("Successfully loaded base model + adapters")
            except Exception as adapter_error:
                print(f"Failed to load base model + adapters: {adapter_error}")
                raise RuntimeError(f"Cannot load model from {model_dir}. Ensure merged model was saved properly.")
        else:
            # Try without quantization as final fallback
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print("Successfully loaded model without quantization")
            except Exception as final_error:
                raise RuntimeError(f"All model loading attempts failed: {final_error}")
    
    # Optimize model for inference
    model.eval()  # Set to evaluation mode
    
    # Clear any existing gradients
    for param in model.parameters():
        param.grad = None
    
    # Test Outlines integration first
    outlines_working = test_outlines_integration(model, tokenizer)
    if not outlines_working:
        print("Outlines integration failed, will use free-form generation only")
    
    # Get unique article IDs from test dataset
    test_article_ids = list(set(item['article_id'] for item in dataset_dict['test']))
    
    results = []
    
    for article_id in test_article_ids:
        try:
            # Check memory before processing each article
            if torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info()
                available_memory = memory_info[0] / 1024**3
                used_memory = (memory_info[1] - memory_info[0]) / 1024**3
                print(f"Processing {article_id}: Memory usage: {used_memory:.2f} GB used, {available_memory:.2f} GB available")
                
                # If less than 2GB available, force garbage collection
                if available_memory < 2.0:
                    print(f"Low memory detected ({available_memory:.2f} GB), clearing cache")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            # Load article text
            article_text = load_article_text(article_id, output_dir)
            
            # Truncate article text if too long to prevent memory issues
            max_article_length = MAX_CHARS_PER_ARTICLE  # Limit article to ~50k characters
            if len(article_text) > max_article_length:
                print(f"Truncating article {article_id} from {len(article_text)} to {max_article_length} characters")
                article_text = article_text[:max_article_length] + "\n[... text truncated ...]"
            
            # Create chat prompt using template
            user_prompt = prompt_text.replace('{article_text}', article_text)
            messages = [
                {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template for inference
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Try structured generation first with retry logic (only if Outlines is available)
            datasets = []
            structured_success = False
            raw_response_text = ""
            
            if OUTLINES_AVAILABLE:
                for attempt in range(1):  # Try up to 1 time to avoid memory issues
                    try:
                        # Check memory before creating Outlines objects
                        if torch.cuda.is_available():
                            memory_info = torch.cuda.mem_get_info()
                            available_memory = memory_info[0] / 1024**3
                            total_memory = memory_info[1] / 1024**3
                            print(f"Memory before Outlines creation: {available_memory:.2f}/{total_memory:.2f} GB")
                            
                            # Prevent 21GB allocation by requiring more available memory
                            # This is the critical fix: check for sufficient memory before Outlines creation
                            memory_required = 10.0  # Conservative estimate for Outlines model creation
                            if available_memory < memory_required:
                                print(f"CRITICAL: Insufficient memory ({available_memory:.2f} GB), need {memory_required:.1f} GB")
                                print(f"Skipping structured generation to prevent 21GB allocation error")
                                break
                        
                        # Try to get cached generator first to avoid recreation
                        model_id = f"{model_dir}_{article_id}"
                        json_generator = get_cached_outlines_generator(model, tokenizer, model_id)
                        
                        if json_generator is None:
                            # MEMORY OPTIMIZATION: Temporarily move base model to CPU to free GPU memory
                            print(f"Moving base model to CPU to create Outlines object...")
                            original_device = next(model.parameters()).device
                            model.cpu()
                            torch.cuda.empty_cache()
                            
                            # Check available memory after CPU offload
                            if torch.cuda.is_available():
                                memory_info = torch.cuda.mem_get_info()
                                available_memory = memory_info[0] / 1024**3
                                print(f"Memory after CPU offload: {available_memory:.2f} GB")
                            
                            # Create fresh Outlines model (this will load model on GPU again)
                            outlines_model = models.from_transformers(model, tokenizer)
                            json_generator = Generator(outlines_model, Response)
                            
                            # Move original model back to GPU after Outlines creation
                            model.to(original_device)
                            print(f"Base model restored to {original_device}")
                        else:
                            print(f"Using cached Outlines generator for {article_id}")
                        
                        print(f"Attempting structured generation for {article_id} (attempt {attempt + 1})")
                        
                        # Check available memory before generation
                        if torch.cuda.is_available():
                            available_memory = torch.cuda.mem_get_info()[0] / 1024**3
                            if available_memory < 2.0:  # Less than 2GB available
                                print(f"Low GPU memory ({available_memory:.2f} GB), clearing cache")
                                torch.cuda.empty_cache()
                        
                        response = json_generator(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.9, repetition_penalty=1.1)
                        response_str = str(response)
                        print(f"Raw response type: {type(response)}")
                        print(f"Raw response length: {len(response_str)} characters")
                        print(f"Raw response preview: {response_str[:200]}{'...' if len(response_str) > 200 else ''}")
                        
                        # Store raw response text
                        raw_response_text = str(response)
                        print(f"Stored raw response length: {len(raw_response_text)} characters")
                        
                        # Handle different response types
                        datasets = []
                        if hasattr(response, 'datasets'):
                            datasets = response.datasets
                        elif isinstance(response, dict) and 'datasets' in response:
                            datasets = response['datasets']
                        elif isinstance(response, list):
                            datasets = response
                        elif isinstance(response, str):
                            # Try to parse string response as JSON
                            try:
                                parsed_response = json.loads(response)
                                if isinstance(parsed_response, list):
                                    datasets = parsed_response
                                elif isinstance(parsed_response, dict) and 'datasets' in parsed_response:
                                    datasets = parsed_response['datasets']
                                else:
                                    print(f"Unexpected parsed response format for {article_id}: {type(parsed_response)}")
                            except json.JSONDecodeError:
                                print(f"Failed to parse string response as JSON for {article_id}: {len(response)} characters")
                                print(f"Response preview: {str(response)[:100]}{'...' if len(str(response)) > 100 else ''}")
                        else:
                            print(f"Unexpected response format for {article_id}: {type(response)}")
                            response_content = str(response)
                            print(f"Response content length: {len(response_content)} characters")
                            print(f"Response content preview: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")
                        
                        print(f"Generated structured response for {article_id}: {len(datasets)} datasets")
                        structured_success = True
                        
                        # Don't delete cached generators, just clear intermediate caches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        break
                        
                    except Exception as gen_error:
                        print(f"Structured generation attempt {attempt + 1} failed for {article_id}: {gen_error}")
                        print(f"Error type: {type(gen_error).__name__}")
                        
                        # Ensure model is moved back to original device on error
                        try:
                            if 'original_device' in locals() and not next(model.parameters()).device == original_device:
                                model.to(original_device)
                                print(f"Model restored to {original_device} after error")
                        except:
                            pass
                        
                        # Clean up any partially created objects (but not cached ones)
                        try:
                            if 'outlines_model' in locals() and json_generator is None:
                                # Only delete if we created a new one (not cached)
                                del outlines_model
                        except:
                            pass
                        
                        # Clear GPU cache on error to prevent memory accumulation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        import gc
                        gc.collect()
                        
                        if attempt == 0:  # Only 1 attempt now
                            print(f"Structured generation attempt failed for {article_id}")
                        continue
            else:
                print(f"Skipping structured generation for {article_id} (Outlines not available)")
            
            # If structured generation failed or returned no datasets, try free-form generation
            if not structured_success:
                try:
                    print(f"Attempting free-form generation for {article_id}")
                    
                    # Tokenize input with aggressive truncation to avoid context length issues
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=MAX_TOKEN_LENGTH,  # Further reduced context to save memory
                        padding=True
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Generate with more conservative parameters to reduce memory usage
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,  # Limit output length
                            temperature=0.1,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            use_cache=False,  # Disable KV cache to save memory
                            low_memory=True,  # Enable low memory mode if available
                            num_beams=1  # Use greedy decoding to save memory
                        )
                    
                    # Decode response
                    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    print(f"Free-form generation for {article_id}: {len(generated_text)} characters")
                    print(f"Free-form response preview: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
                    
                    # Store raw response text for free-form generation
                    raw_response_text = generated_text
                    
                    # Clean up generation tensors immediately
                    del outputs
                    del inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Use the new parsing function
                    freeform_datasets = parse_model_response(generated_text)
                    # Only use freeform results if structured generation failed completely
                    if not structured_success:
                        datasets = freeform_datasets
                        print(f"Using free-form results: {len(datasets)} datasets found")
                        
                except Exception as fallback_error:
                    print(f"Free-form generation also failed for {article_id}: {fallback_error}")
                    # Clear GPU cache on fallback error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not structured_success:
                        datasets = []
            
            response = Response(datasets=datasets)
            results.append({
                'article_id': article_id,
                'response': response
            })
            
            # Write raw response to file
            write_response_to_file(article_id, raw_response_text, output_dir)
            
            print(f"Processed article {article_id}: found {len(datasets)} datasets")
            if len(datasets) > 0:
                dataset_names = []
                for d in datasets[:3]:  # Show first 3 datasets
                    if hasattr(d, 'dataset_name'):
                        dataset_names.append(d.dataset_name)
                    elif isinstance(d, dict) and 'dataset_name' in d:
                        dataset_names.append(d['dataset_name'])
                    else:
                        dataset_names.append(str(d))
                print(f"Sample datasets: {dataset_names}{'...' if len(datasets) > 3 else ''}")
            
            # Clear any accumulated gradients and GPU cache after each article
            for param in model.parameters():
                param.grad = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"GPU memory cleared after {article_id}. Available: {available_memory:.2f} GB")
            
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            # Clear GPU cache even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create empty response for failed articles
            datasets = []
            response = Response(datasets=datasets)
            results.append({
                'article_id': article_id,
                'response': response
            })
            
            # Write empty raw response to file even for failed articles
            write_response_to_file(article_id, "", output_dir)
    
    # Clear Outlines cache at the end of inference to free memory
    clear_outlines_cache()
    
    return results


def run_inference_simple(dataset_dict, output_dir, model_dir):
    """
    Run inference using only free-form generation without Outlines structured generation.
    This is a fallback method when structured generation fails.
    
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
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"Loading trained model from {model_path}")
    
    # Load model with error handling
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Error loading model with quantization: {e}")
        print("Attempting to load model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get unique article IDs from test dataset
    test_article_ids = list(set(item['article_id'] for item in dataset_dict['test']))
    
    results = []
    
    for article_id in test_article_ids:
        try:
            # Load article text
            article_text = load_article_text(article_id, output_dir)
            
            # Create chat prompt using template
            user_prompt = prompt_text.replace('{article_text}', article_text)
            messages = [
                {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template for inference
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=MAX_TOKEN_LENGTH
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Generated text for {article_id}: {len(generated_text)} characters")
            print(f"Response preview: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
            
            # Use the new parsing function
            datasets = parse_model_response(generated_text)
            
            response = Response(datasets=datasets)
            results.append({
                'article_id': article_id,
                'response': response
            })
            
            # Write raw response to file
            write_response_to_file(article_id, generated_text, output_dir)
            
            print(f"Processed article {article_id}: found {len(datasets)} datasets")
            if len(datasets) > 0:
                dataset_names = []
                for d in datasets[:3]:  # Show first 3 datasets
                    if hasattr(d, 'dataset_name'):
                        dataset_names.append(d.dataset_name)
                    elif isinstance(d, dict) and 'dataset_name' in d:
                        dataset_names.append(d['dataset_name'])
                    else:
                        dataset_names.append(str(d))
                print(f"Sample datasets: {dataset_names}{'...' if len(datasets) > 3 else ''}")
            
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            datasets = []
            response = Response(datasets=datasets)
            results.append({
                'article_id': article_id,
                'response': response
            })
            
            # Write empty raw response to file even for failed articles
            write_response_to_file(article_id, "", output_dir)
    
    return results


def write_response_to_file(article_id, raw_response_text, output_dir):
    """
    Write the raw model response to a file in the specified output directory.
    
    Args:
        article_id (str): The article identifier
        raw_response_text (str): The raw response text from the model
        output_dir (str): Directory to write response files
    """
    response_file = Path(output_dir) / f"{article_id}.response"
    
    # Write raw response to file
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(raw_response_text)
    
    print(f"Raw response written to {response_file} ({len(raw_response_text)} characters)")


