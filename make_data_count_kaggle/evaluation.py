import pandas as pd

def calculate_f1_score(ground_truth_df, predictions_df):
    """
    Calculates the micro F1 score between predicted and ground truth citations.
    A citation is a tuple of (article_id, dataset_id, type).
    
    Args:
        ground_truth_df (pd.DataFrame): DataFrame with ground truth. 
                                        Expected columns: 'article_id', 'dataset_id', 'dataset_label'.
        predictions_df (pd.DataFrame): DataFrame with predictions.
                                       Expected columns: 'article_id', 'dataset_id', 'type'.
                                       
    Returns:
        tuple: f1_score, precision, recall, tp, fp, fn
    """
    # Ground truth citations
    true_citations = set()
    for _, row in ground_truth_df.iterrows():
        true_citations.add((row['article_id'], row['dataset_id'], row['type']))

    # Predicted citations
    pred_citations = set()
    for _, row in predictions_df.iterrows():
        pred_citations.add((row['article_id'], row['dataset_id'], row['type']))

    # Calculate TP, FP, FN
    tp = len(true_citations.intersection(pred_citations))
    fp = len(pred_citations - true_citations)
    fn = len(true_citations - pred_citations)

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall, tp, fp, fn
