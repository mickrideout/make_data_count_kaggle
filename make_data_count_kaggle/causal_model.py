import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import torch

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

## Response


"""


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
        Dataset: HuggingFace dataset with article_id and completion data
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
    
    return Dataset.from_list(training_data)


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
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"  # Use smaller model for stability
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 to avoid precision issues
        device_map=None,  # Force CPU mode for stability
        low_cpu_mem_usage=True
    )
    
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        num_train_epochs=1,  # Reduce epochs for initial testing
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=50,  # Reduce warmup steps
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,  # Disable gradient checkpointing
        fp16=False,
        max_grad_norm=1.0,  # Add gradient clipping
        report_to=None,  # Disable wandb logging
    )
    
    # Create SFTTrainer
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