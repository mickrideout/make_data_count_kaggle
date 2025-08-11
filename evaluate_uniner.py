#!/usr/bin/env python3
"""
Evaluation script for comparing submission.csv to train_labels.csv and calculating F1-score.

This script evaluates model predictions against ground truth labels using the same
evaluation logic as the training pipeline.

Usage:
    python evaluate_uniner.py <submission_file> <ground_truth_file> [output_dir]

Example:
    python evaluate_uniner.py submission.csv train_labels.csv ./evaluation_output
"""

import sys
import os
import pandas as pd
from make_data_count_kaggle.evaluation import calculate_f1_score


def validate_dataframe(df, filename, required_columns):
    """
    Validates that a DataFrame has the required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        filename (str): Name of the file for error messages
        required_columns (list): List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{filename} is missing required columns: {missing_columns}")


def load_and_validate_files(submission_path, ground_truth_path):
    """
    Loads and validates submission and ground truth CSV files.
    
    Args:
        submission_path (str): Path to submission CSV file
        ground_truth_path (str): Path to ground truth CSV file
        
    Returns:
        tuple: (submission_df, ground_truth_df)
        
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If files have incorrect format
    """
    # Check if files exist
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    # Load CSV files
    try:
        submission_df = pd.read_csv(submission_path)
        ground_truth_df = pd.read_csv(ground_truth_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Error reading CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading files: {e}")
    
    # Validate required columns
    required_columns = ['article_id', 'dataset_id', 'type']
    validate_dataframe(submission_df, submission_path, required_columns)
    validate_dataframe(ground_truth_df, ground_truth_path, required_columns)
    
    # Filter out rows with 'Missing' values from ground truth
    original_gt_size = len(ground_truth_df)
    ground_truth_df = ground_truth_df[
        (ground_truth_df['dataset_id'] != 'Missing') & 
        (ground_truth_df['type'] != 'Missing')
    ]
    filtered_gt_size = len(ground_truth_df)
    
    if original_gt_size != filtered_gt_size:
        print(f"Filtered out {original_gt_size - filtered_gt_size} rows with 'Missing' values from ground truth")
    
    return submission_df, ground_truth_df


def main():
    """
    Main function to run the evaluation.
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python evaluate_uniner.py <submission_file> <ground_truth_file> [output_dir]")
        print("\nExample:")
        print("    python evaluate_uniner.py submission.csv train_labels.csv ./evaluation_output")
        sys.exit(1)
    
    submission_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) == 4 else "./evaluation_output"
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and validate files
        print(f"Loading submission file: {submission_path}")
        print(f"Loading ground truth file: {ground_truth_path}")
        submission_df, ground_truth_df = load_and_validate_files(submission_path, ground_truth_path)
        
        print(f"\nSubmission shape: {submission_df.shape}")
        print(f"Ground truth shape: {ground_truth_df.shape}")
        
        # Calculate F1 score using existing evaluation function
        print(f"\nCalculating F1 score...")
        f1, precision, recall, tp, fp, fn = calculate_f1_score(
            ground_truth_df, submission_df, output_dir
        )
        
        # Display results
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"F1 Score:    {f1:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"{'='*50}")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"{'='*50}")
        
        # Display additional statistics
        total_predictions = len(submission_df)
        total_ground_truth = len(ground_truth_df)
        print(f"\nAdditional Statistics:")
        print(f"Total predictions:    {total_predictions}")
        print(f"Total ground truth:   {total_ground_truth}")
        print(f"Correct predictions:  {tp}")
        
        # Information about output files
        print(f"\nDetailed analysis files saved to: {output_dir}")
        print(f"- tp.csv: True positive predictions")
        print(f"- fp.csv: False positive predictions")  
        print(f"- fn.csv: False negative predictions")
        
        return f1
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()