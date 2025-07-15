import pandas as pd

CANDIDATE_CSV_FILE = "candidate_dataset.csv"
SUBMISSION_FILE = "submission.csv"
CANDIDATE_HEADER = ["row_id", "article_id", "dataset_id", "type"]

def dummy_classifier(input_dir, output_dir):
    """
    Dummy classifier that classifies all candidates as "Primary".
    """
    print("Performing dummy classification")

    input_df = pd.read_csv(f"{output_dir}/{CANDIDATE_CSV_FILE}")
    output_df = pd.DataFrame(columns=CANDIDATE_HEADER)
    
    for _, row in input_df.iterrows():
        if row["dataset_id"] not in output_df["dataset_id"].values:
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
            output_df.loc[len(output_df) - 1, "type"] = "Primary"
            output_df.loc[len(output_df) - 1, "row_id"] = len(output_df) - 1
            output_df.drop(columns=["candidate_matcher", "candidate_text"], inplace=True)
    

    output_df.to_csv(f"{output_dir}/{SUBMISSION_FILE}", index=False)
    return output_df
