import os
import sys
import pandas as pd
import mlflow

from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_text, convert_xmls_to_text, decompose_text_to_lines, decompose_train_labels, create_dataset_for_training, train_test_split
from make_data_count_kaggle.evaluation import calculate_f1_score
from make_data_count_kaggle.universal_ner import generate_uniner_dataset


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
   

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
    train_df.to_csv(f"{output_directory}/train_dataset.csv", index=False, escapechar='\\', quoting=1)
    test_df.to_csv(f"{output_directory}/test_dataset.csv", index=False, escapechar='\\', quoting=1)





