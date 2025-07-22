import os
import sys
import pandas as pd

from make_data_count_kaggle.causal_model import run_inference, run_inference_simple
from make_data_count_kaggle.data_preprocessing import convert_labels_csv_to_json, convert_pdfs_to_text, convert_xmls_to_text, create_huggingface_dataset, decompose_text_to_paragraphs, decompose_train_labels
from make_data_count_kaggle.dataset_matching import basic_matching, create_empty_candidate_dataset
from make_data_count_kaggle.dataset_classification import dummy_classifier


def main(input_directory, output_directory, model_dir):
    # Dataset preprocessing
    decompose_train_labels(input_directory, output_directory)
    convert_labels_csv_to_json(output_directory)
    convert_xmls_to_text(f"{input_directory}/train", output_directory)
    convert_pdfs_to_text(f"{input_directory}/train", output_directory)
    decompose_text_to_paragraphs(output_directory)
    dataset_dict = create_huggingface_dataset(output_directory)

    # Causal model training
    SUBMISSION_FILE = "submission.csv"

    # Causal model inference with fallback
    try:
        print("Attempting inference with structured generation...")
        inference_results = run_inference(dataset_dict, output_directory, model_dir)
    except Exception as e:
        print(f"Structured inference failed: {e}")
        print("Falling back to simple inference...")
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
