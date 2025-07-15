import os
import sys
import pandas as pd

from make_data_count_kaggle.dataset_classification import dummy_classifier
from make_data_count_kaggle.dataset_matching import basic_matching, create_empty_candidate_dataset
from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_markdown, decompose_text_to_paragraphs
from make_data_count_kaggle.evaluation import calculate_f1_score


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    # Dataset preprocessing
    convert_pdfs_to_markdown(f"{input_directory}/train", output_directory)
    decompose_text_to_paragraphs(output_directory)

    # Candidate generation
    create_empty_candidate_dataset(output_directory)
    basic_matching(output_directory, output_directory)

    # Candidate classification
    output_df = dummy_classifier(output_directory, output_directory)

    # Evaluation
    ground_truth_df = pd.read_csv(f"{input_directory}/train_labels.csv")
    f1, precision, recall, tp, fp, fn = calculate_f1_score(ground_truth_df, output_df, output_directory)

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")



