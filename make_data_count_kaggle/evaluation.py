import pandas as pd

def calculate_f1_score(ground_truth_df, predictions_df, output_dir):
    """
    Calculates the micro F1 score between predicted and ground truth citations.
    A citation is a tuple of (article_id, dataset_id, type).
    
    Args:
        ground_truth_df (pd.DataFrame): DataFrame with ground truth. 
                                        Expected columns: 'article_id', 'dataset_id', 'type'.
        predictions_df (pd.DataFrame): DataFrame with predictions.
                                       Expected columns: 'article_id', 'dataset_id', 'type'.
        output_dir (str): Directory to save TP, FP, FN records.
                                       
    Returns:
        tuple: f1_score, precision, recall, tp, fp, fn
    """
    # Use a merge to find TP, FP, FN
    merged_df = ground_truth_df.merge(
        predictions_df, 
        on=['article_id', 'dataset_id', 'type'], 
        how='outer', 
        indicator=True
    )

    # True Positives (present in both)
    tp_df = merged_df[merged_df['_merge'] == 'both']
    tp = len(tp_df)

    # False Positives (present in predictions only)
    fp_df = merged_df[merged_df['_merge'] == 'right_only']
    fp = len(fp_df)

    # False Negatives (present in ground truth only)
    fn_df = merged_df[merged_df['_merge'] == 'left_only']
    fn = len(fn_df)

    # Save records to CSV
    tp_df[['article_id', 'dataset_id', 'type']].to_csv(f"{output_dir}/tp.csv", index=False)
    fp_df[['article_id', 'dataset_id', 'type']].to_csv(f"{output_dir}/fp.csv", index=False)
    fn_df[['article_id', 'dataset_id', 'type']].to_csv(f"{output_dir}/fn.csv", index=False)

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall, tp, fp, fn