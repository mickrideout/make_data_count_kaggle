#!/usr/bin/env python3

import os
# Set environment variables for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import snapshot_download

#deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#microsoft/Phi-4-mini-instruct
#microsoft/Phi-4-reasoning
#prithivMLmods/Galactic-Qwen-14B-Exp2
#ibm-granite/granite-3.2-8b-instruct
def load_model(model_name, model_dir):
    """Load the Hugging Face model and tokenizer with memory optimization."""

    if not os.path.exists(model_dir):
        snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

    
    # Check if CUDA is available and has sufficient memory
    if torch.cuda.is_available():
        try:
            # Load model with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",  # Automatically handle device placement
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                trust_remote_code=True,
                local_files_only=True
            )
            print("Model loaded on GPU with memory optimization")
        except torch.cuda.OutOfMemoryError:
            print("GPU memory insufficient, falling back to CPU")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True
            )
    else:
        print("CUDA not available, using CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True
        )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def read_article_file(file_path):
    """Read article text from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def run_inference(model, tokenizer, prompt):
    """Run inference on the model with the given prompt using chat template."""
    # Clear any cached memory
    torch.cuda.empty_cache()
    
    try:
        # Try chat template approach first
        messages = [
            {"role": "system", "content": "You are an expert in extracting and categorizing dataset mentions from research papers and policy documents."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
    except (ImportError, AttributeError, Exception) as e:
        print(f"Chat template not available, using direct tokenization: {e}")
        # Fallback to direct tokenization
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Extract only the generated part (excluding input)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Clear memory after generation
    torch.cuda.empty_cache()
    
    return response


def main():
    parser = argparse.ArgumentParser(description='Run Hugging Face model inference with article text')
    parser.add_argument('article_file', help='Path to file containing article text')
    parser.add_argument('--model', help='Hugging Face model to use')
    parser.add_argument('--model-dir', help='Model directory to use')
    
    args = parser.parse_args()
    
    # Hardcoded prompt template with {article_text} placeholder
    prompt_template = """Your task is to **identify and extract all valid dataset mentions** and ensure their type is correctly classified. You to read, parse and then perform the extraction of the text in the ## **Text** section below. Provide your reponse in json format defined in the section ### **Extraction Schema**


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
   - "The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395." → `"Secondary"`.

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

    
    # Read article text from file
    article_text = read_article_file(args.article_file)
    
    # Replace the placeholder with actual article text
    prompt = prompt_template.replace('{article_text}', article_text)
    
    print("Loading model...")
    model, tokenizer = load_model(args.model, args.model_dir)
    
    print("Running inference...")
    response = run_inference(model, tokenizer, prompt)
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(response)


if __name__ == "__main__":
    main()