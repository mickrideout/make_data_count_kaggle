import os
import sys
import pandas as pd

# Set PyTorch CUDA memory management environment variables for better memory handling
# Prevent large allocations and improve memory fragmentation handling
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'  # Disable memory caching for more predictable behavior

from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_text, convert_xmls_to_text, decompose_text_to_lines, decompose_train_labels, create_dataset, train_test_split
from make_data_count_kaggle.evaluation import calculate_f1_score
from make_data_count_kaggle.universal_ner import generate_dataset


def main(input_directory, output_directory, model_dir):
    print("Starting inference pipeline with memory optimizations...")
    
    # Import torch for memory management
    import torch
    import gc
    
    # Dataset preprocessing
    decompose_train_labels(input_directory, output_directory)
    convert_xmls_to_text(f"{input_directory}/train", output_directory)
    convert_pdfs_to_text(f"{input_directory}/train", output_directory)
    decompose_text_to_lines(output_directory)
    create_dataset(input_directory, output_directory)
    
    # Clear any Python garbage before model loading
    gc.collect()
    
    # Additional memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Check available memory before proceeding
        memory_info = torch.cuda.mem_get_info()
        available_memory = memory_info[0] / 1024**3
        total_memory = memory_info[1] / 1024**3
        print(f"Pre-inference memory check: {available_memory:.2f}/{total_memory:.2f} GB available")
        
        if available_memory < 5.0:
            print(f"WARNING: Low GPU memory ({available_memory:.2f} GB). Consider using smaller model or batch size.")

    # Causal model training
    SUBMISSION_FILE = "submission.csv"

    # Causal model inference with fallback
    try:
        print("Attempting inference with structured generation...")
        inference_results = run_inference(dataset_dict, output_directory, model_dir)
    except Exception as e:
        print(f"Structured inference failed: {e}")
        print("Falling back to simple inference...")
        # Clear memory before fallback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_info = torch.cuda.mem_get_info()
            available_memory = memory_info[0] / 1024**3
            print(f"Memory after clearing cache: {available_memory:.2f} GB available")
        gc.collect()
        inference_results = run_inference_simple(dataset_dict, output_directory, model_dir)
    
    # Convert inference results to output dataframe for evaluation
    output_data = []
    for result in inference_results:
        article_id = result['article_id']
        response = result['response']
        for dataset in response.datasets:
            output_data.append({
                'article_id': article_id,
                'dataset_id': dataset.dataset_name,
                'type': dataset.dataset_type
            })
    
    # Create DataFrame with proper columns even if empty
    if output_data:
        output_df = pd.DataFrame(output_data)
    else:
        output_df = pd.DataFrame(columns=['article_id', 'dataset_id', 'type'])

    output_df.to_csv(f"{output_directory}/{SUBMISSION_FILE}", index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python infer.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_dir = sys.argv[3]

    main(input_directory, output_directory, model_dir)
