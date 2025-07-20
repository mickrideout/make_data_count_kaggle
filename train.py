import os
import sys
import pandas as pd
import mlflow

from make_data_count_kaggle.dataset_classification import dummy_classifier
from make_data_count_kaggle.dataset_matching import basic_matching, create_empty_candidate_dataset
from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_text, convert_xmls_to_text, decompose_text_to_paragraphs, decompose_train_labels, convert_labels_csv_to_json, create_huggingface_dataset
from make_data_count_kaggle.evaluation import calculate_f1_score
from make_data_count_kaggle.causal_model import train_causal_model, run_inference


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    with mlflow.start_run():
        mlflow.log_param("input_directory", input_directory)
        mlflow.log_param("output_directory", output_directory)

        # Dataset preprocessing
        decompose_train_labels(input_directory, output_directory)
        convert_labels_csv_to_json(output_directory)
        convert_xmls_to_text(f"{input_directory}/train", output_directory)
        convert_pdfs_to_text(f"{input_directory}/train", output_directory)
        decompose_text_to_paragraphs(output_directory)
        dataset_dict = create_huggingface_dataset(output_directory)

        # Causal model training
        causal_model_dir = f"{output_directory}/causal_model"
        train_causal_model(dataset_dict, output_directory, causal_model_dir)

        # Causal model inference
        inference_results = run_inference(dataset_dict, output_directory, causal_model_dir)
        
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
        
        print(f"Generated {len(output_df)} predictions from {len(inference_results)} articles")
        print(f"Output DataFrame columns: {list(output_df.columns)}")
        print(f"Output DataFrame shape: {output_df.shape}")
        if len(output_df) > 0:
            print(f"Sample output data:\n{output_df.head()}")

        # Evaluation
        ground_truth_df = pd.read_csv(f"{input_directory}/train_labels.csv")
        f1, precision, recall, tp, fp, fn = calculate_f1_score(ground_truth_df, output_df, output_directory)

        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("true_positives", tp)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)





