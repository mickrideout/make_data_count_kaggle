import pandas as pd
import json


def truncate_text(text, max_chars=8000):
    """
    Truncate text to a reasonable length to prevent tokenization issues.
    
    Args:
        text (str): Input text to truncate
        max_chars (int): Maximum number of characters to keep
        
    Returns:
        str: Truncated text
    """
    if not isinstance(text, str):
        return ""
    
    if len(text) <= max_chars:
        return text
    
    # Truncate to max_chars and add ellipsis
    return text[:max_chars] + "..."


def create_json(text, dataset_id, type):
    def is_missing(val):
        if val is None:
            return True
        if isinstance(val, float) and pd.isna(val):
            return True
        if isinstance(val, str) and val.strip().lower() == "nan":
            return True
        return False

    # Truncate text to prevent tokenization issues
    truncated_text = truncate_text(text)

    return {
        "conversations": [
            {
                "from": "human",
                "value": truncated_text
            },
            {
                "from": "gpt",
                "value": "I've read this text."
            },
            {
                "from": "human",
                "value": "What describes the datasets that are mentioned in this text?"
            },
            {
                "from": "gpt",
                "value": "[]" if is_missing(dataset_id) else f"[\"{dataset_id}\"]"
            },
            {
                "from": "human",
                "value": "What describes the types of dataset sources that are mentioned in this text? (Primary if generated / constructed, Secondary if provided / utilised)"
            },
            {
                "from": "gpt",
                "value": "[]" if is_missing(type) else f"[\"{type}\"]"
            },
        ]
    }

# create a function, generate_dataset, which takes an pandas df as an argument (which has the columns text, dataset_id and type) and an output file path, iterate over the rows and use create_json to generate the json for that row. add all rows to an array and write the json to the output file
def generate_uniner_dataset(df, output_file_path):
    """
    Generate a dataset from a pandas DataFrame and write to JSON file.
    
    Args:
        df: pandas DataFrame with columns 'text', 'dataset_id', and 'type'
        output_file_path: path to the output JSON file
    """
    dataset_array = []
    
    for _, row in df.iterrows():
        json_item = create_json(row['text'], row['dataset_id'], row['type'])
        dataset_array.append(json_item)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_array, f, indent=2, ensure_ascii=False)

