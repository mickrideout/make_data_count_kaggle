import os
import sys
import pandas as pd
import mlflow

from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_text, convert_xmls_to_text, decompose_text_to_lines, decompose_train_labels, create_dataset_for_training, train_test_split
from make_data_count_kaggle.evaluation import calculate_f1_score
from make_data_count_kaggle.universal_ner import generate_dataset


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_dir = sys.argv[3]
    
    with mlflow.start_run():
        mlflow.log_param("input_directory", input_directory)
        mlflow.log_param("output_directory", output_directory)
        mlflow.log_param("model_dir", model_dir)
        # Dataset preprocessing
        decompose_train_labels(input_directory, output_directory)
        convert_xmls_to_text(f"{input_directory}/train", output_directory)
        convert_pdfs_to_text(f"{input_directory}/train", output_directory)
        decompose_text_to_lines(output_directory)
        create_dataset_for_training(input_directory, output_directory)
        train_df, test_df = train_test_split(f"{output_directory}/dataset.csv")
        print(train_df.head())
        print(test_df.head())
        print(train_df.shape)
        print(test_df.shape)
        print(train_df['dataset_id'].value_counts())
        print(test_df['dataset_id'].value_counts())
        generate_dataset(train_df, f"{output_directory}/train_dataset.json")
        generate_dataset(test_df, f"{output_directory}/test_dataset.json")



        exit()
        # Causal model training


        # Causal model inference
        inference_results = run_inference(dataset_dict, output_directory, model_dir)
        
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





