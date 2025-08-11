import os
import glob
import pickle
import signal
import fitz
from io import StringIO
import xml.etree.ElementTree as ET
import re
import pandas as pd
from pathlib import Path
import concurrent.futures
from functools import partial
import json
import random
from typing import List
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import numpy as np

DATASET_FILE = "dataset.csv"
TRAINING_LABELS_FILE = "train_labels.csv"

def clean_text(text):
    """
    Clean up text by removing references section.
    
    Args:
        text (str): Raw text from PyMuPDF
        
    Returns:
        str: Cleaned text with references section removed
    """
    # Remove references section
    # TODO: Implement this
    
    return text


def _convert_pdf_to_text_worker(pdf_file, output_dir):
    """
    Worker function to convert a single PDF to text using PyMuPDF with a 2-minute timeout.
    """
    pdf_path = Path(pdf_file)
    output_path = Path(output_dir)
    filename = pdf_path.stem
    output_file = output_path / f"{filename}.txt"

    if output_file.exists():
        return f"Skipping {pdf_path.name} - {output_file.name} already exists"

    # Define a handler for the timeout
    def handler(signum, frame):
        raise TimeoutError("PDF conversion timed out after 2 minutes")

    # Set the signal handler and a 2-minute (120 seconds) alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(120)

    try:
        # Use PyMuPDF to extract text
        text_parts = []
        
        with fitz.open(str(pdf_path)) as pdf:
            for page in pdf:
                # Extract text from the page
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
                
                # Extract tables from the page using table finder
                tables = page.find_tables()
                for table in tables:
                    if table and len(table.extract()) > 0:
                        # Convert table to text representation
                        table_data = table.extract()
                        table_text = '\n'.join(['\t'.join([str(cell) if cell else '' for cell in row]) for row in table_data])
                        if table_text.strip():
                            text_parts.append(table_text)
        
        # Combine all text parts
        text = '\n\n'.join(text_parts)
        
        if text.strip():
            cleaned_text = clean_text(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            return f"Successfully converted {pdf_path.name}"
        else:
            return f"No text content found in {pdf_path.name}"
    except TimeoutError as e:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")
        return f"Error converting {pdf_path.name}: {str(e)}"
    except Exception as e:
        return f"Error converting {pdf_path.name}: {str(e)}"
    finally:
        # Disable the alarm
        signal.alarm(0)


def convert_pdfs_to_text(input_dir, output_dir):
    """
    Convert all PDF files in input_dir to text files in output_dir using PyMuPDF in parallel.
    
    Args:
        input_dir (str): Directory containing PDF files to convert
        output_dir (str): Directory where text files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = glob.glob(str(input_path / "**/*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function to pass the output_dir to the worker
        convert_func = partial(_convert_pdf_to_text_worker, output_dir=output_dir)
        
        # Process files in parallel
        results = executor.map(convert_func, pdf_files)
        
        # Output results
        for result in results:
            print(result)
            
    print(f"Conversion complete. Text files saved to {output_dir}")


def _convert_xml_to_text_worker(xml_file, output_dir):
    """
    Worker function to convert a single XML to text.
    """
    xml_path = Path(xml_file)
    output_path = Path(output_dir)
    filename = xml_path.stem
    output_file = output_path / f"{filename}.txt"

    if output_file.exists():
        return f"Skipping {xml_path.name} - {output_file.name} already exists"

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text = " ".join(root.itertext())
        
        if text.strip():
            cleaned_text = clean_text(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            return f"Successfully converted {xml_path.name}"
        else:
            return f"No text content found in {xml_path.name}"
    except Exception as e:
        return f"Error converting {xml_path.name}: {str(e)}"


def convert_xmls_to_text(input_dir, output_dir):
    """
    Convert all XML files in input_dir to text files in output_dir in parallel.
    
    Args:
        input_dir (str): Directory containing XML files to convert
        output_dir (str): Directory where text files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    
    xml_files = glob.glob(str(input_path / "**/*.xml"), recursive=True)
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to convert")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function to pass the output_dir to the worker
        convert_func = partial(_convert_xml_to_text_worker, output_dir=output_dir)
        
        # Process files in parallel
        results = executor.map(convert_func, xml_files)
        
        # Output results
        for result in results:
            print(result)
            
    print(f"Conversion complete. Text files saved to {output_dir}")


def _decompose_text_worker(text_file, output_dir):
    """
    Worker function to decompose a single text file into lines using advanced NLP methods.
    """
    text_path = Path(text_file)
    output_path = Path(output_dir)
    filename = text_path.stem
    pickle_file = os.path.join(output_dir, f"{filename}.pkl")

    if os.path.exists(pickle_file):
        return f"Skipping {text_path.name} - {os.path.basename(pickle_file)} already exists"

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use context window approach for text decomposition
        lines = decompose_text_with_context_window(content)
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(lines, f)
            
        return f"Successfully decomposed {text_path.name} into {len(lines)} lines"
    except Exception as e:
        return f"Error decomposing {text_path.name}: {str(e)}"


def decompose_text_with_context_window(text: str, context_window: int = 1024, overlap: int = 100) -> List[str]:
    """
    Decompose text into chunks based on configurable context window size with overlap.
    
    Args:
        text (str): Input text to decompose
        context_window (int): Maximum size of each chunk in characters (default: 1024)
        overlap (int): Number of characters to overlap between chunks (default: 100)
        
    Returns:
        List[str]: List of text chunks with specified context window size and overlap
    """
    if not text or len(text) <= context_window:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position for current chunk
        end = start + context_window
        
        # If this is the last chunk and it would be smaller than overlap,
        # just extend the previous chunk to include remaining text
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a good break point (sentence end or paragraph break)
        chunk_text = text[start:end]
        
        # Look for sentence boundaries near the end of the chunk
        sentence_breaks = []
        for i, char in enumerate(chunk_text):
            if char in '.!?' and i < len(chunk_text) - 1:
                next_char = chunk_text[i + 1] if i + 1 < len(chunk_text) else ''
                if next_char in ' \n\t' or i == len(chunk_text) - 1:
                    sentence_breaks.append(i + 1)
        
        # Find the best break point in the last quarter of the chunk
        best_break = None
        quarter_point = len(chunk_text) * 3 // 4
        
        for break_point in reversed(sentence_breaks):
            if break_point >= quarter_point:
                best_break = start + break_point
                break
        
        # If no good sentence break found, use the full context window
        if best_break is None:
            best_break = end
        
        chunks.append(text[start:best_break])
        
        # Move start position with overlap
        start = best_break - overlap
        
        # Ensure we don't go backwards
        if start < 0:
            start = 0
    
    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk.strip()) > 20]


def decompose_text_to_lines(output_dir):
    """
    Decompose all text in output_dir into lines and filter for lines longer than 10 characters,
    then save the line arrays as pickle files in parallel.
    
    Args:
        output_dir (str): Directory containing text files to decompose
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    text_files = glob.glob(str(output_path / "*.txt"), recursive=False)
    
    if not text_files:
        print(f"No text files found in {output_dir}")
        return
    
    print(f"Found {len(text_files)} text files to decompose")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function to pass the output_dir to the worker
        decompose_func = partial(_decompose_text_worker, output_dir=output_dir)
        
        # Process files in parallel
        results = executor.map(decompose_func, text_files)
        
        # Output results
        for result in results:
            print(result)
            
    print(f"Decomposition complete. Pickle files saved to {output_dir}")


def decompose_train_labels(input_dir, output_dir):
    """
    Decompose train_labels.csv into training_labels.csv and testing_labels.csv
    based on whether the article_id exists in train/PDF or test/PDF folders.
    
    Args:
        input_dir (str): Directory containing train_labels.csv and train/test folders
        output_dir (str): Directory where training_labels.csv and testing_labels.csv will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Read the train_labels.csv file
    train_labels_file = input_path / "train_labels.csv"
    if not train_labels_file.exists():
        raise ValueError(f"train_labels.csv not found in {input_dir}")
    
    df = pd.read_csv(train_labels_file)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists for training and testing labels
    training_rows = []
    testing_rows = []
    
    for _, row in df.iterrows():
        # Skip rows with Missing in dataset_id or type columns
        if row['dataset_id'] == 'Missing' or row['type'] == 'Missing':
            continue
            
        article_id = row['article_id']
        
        # Check if PDF exists in train folder
        train_pdf_path = input_path / "train" / "PDF" / f"{article_id}.pdf"
        test_pdf_path = input_path / "test" / "PDF" / f"{article_id}.pdf"
        
        if train_pdf_path.exists():
            training_rows.append(row)
        elif test_pdf_path.exists():
            testing_rows.append(row)
    
    # Create DataFrames and save to CSV
    if training_rows:
        training_df = pd.DataFrame(training_rows)
        training_output = output_path / "training_labels.csv"
        training_df.to_csv(training_output, index=False)
        print(f"Created training_labels.csv with {len(training_rows)} rows")
    else:
        print("No training labels found")
    
    if testing_rows:
        testing_df = pd.DataFrame(testing_rows)
        testing_output = output_path / "testing_labels.csv"
        testing_df.to_csv(testing_output, index=False)
        print(f"Created testing_labels.csv with {len(testing_rows)} rows")
    else:
        print("No testing labels found")


def _process_pkl_file_worker(pkl_file, df):
    """
    Worker function to process a single pickle file and extract matching rows.
    
    Args:
        pkl_file (Path): Path to the pickle file
        df (pd.DataFrame): DataFrame containing dataset_id and type information
        
    Returns:
        list: List of dictionaries containing text, dataset_id, and type
    """
    output_rows = []
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            for text in data:
                found_match = False
                for _, row in df.iterrows():
                    # Convert dataset_id to string and handle NaN values
                    dataset_id = row["dataset_id"]
                    if pd.isna(dataset_id):
                        dataset_id = ""
                    else:
                        dataset_id = str(dataset_id)
                    
                    # Only check if dataset_id is not empty
                    if dataset_id and dataset_id in text:
                        output_rows.append({
                            "text": text,
                            "dataset_id": dataset_id,
                            "article_id": row["article_id"],
                            "type": row["type"]
                        })
                        found_match = True
                        break
                if not found_match:
                    output_rows.append({
                        "text": text,
                        "dataset_id": "",
                        "article_id": row["article_id"],
                        "type": ""
                    })
        return output_rows
    except Exception as e:
        print(f"Error processing {pkl_file}: {str(e)}")
        return []


def create_dataset_for_training(input_dir, output_dir):
    """
    Convert training_labels.csv to a single JSON file.
    
    Args:
        input_dir (str): Directory containing training_labels.csv
        output_dir (str): Directory to save the output dataset.json
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    dataset_file = os.path.join(output_dir, DATASET_FILE)
    
    if os.path.exists(dataset_file):
        print(f"Dataset file already exists: {dataset_file}")
        return pd.read_csv(dataset_file)
    
    # Load the training labels CSV
    training_labels_file = os.path.join(input_dir, TRAINING_LABELS_FILE)

    if not os.path.exists(training_labels_file):
        raise ValueError(f"training_labels.csv not found in {input_dir}")
    
    # Read the CSV file
    df = pd.read_csv(training_labels_file)
    
    print(f"Loaded {len(df)} rows from training_labels.csv")

    df = df[(df['dataset_id'] != 'Missing') & (df['type'] != 'Missing')]

    pkl_files = list(Path(output_dir).glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files to process")
    
    output_rows = []
    
    # Process pickle files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create partial function with df parameter
        worker_func = partial(_process_pkl_file_worker, df=df)
        
        # Submit all pickle files for processing
        future_to_file = {executor.submit(worker_func, pkl_file): pkl_file for pkl_file in pkl_files}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            pkl_file = future_to_file[future]
            try:
                result = future.result()
                output_rows.extend(result)
                print(f"Processed {pkl_file.name}: {len(result)} rows")
            except Exception as e:
                print(f"Error processing {pkl_file.name}: {str(e)}")
    
    output_df = pd.DataFrame(output_rows, columns=["article_id", "dataset_id", "type", "text"])
    print(f"Total rows processed: {len(output_df)}")
    
    # Write dataset JSON file

    output_df.to_csv(dataset_file, index=False, escapechar='\\', quoting=1)
    print(f"Created dataset.csv with {len(output_df)} rows")
    
    return output_df


def _process_pkl_file_for_inference_worker(pkl_file):
    """
    Worker function to process a single pickle file for inference.
    
    Args:
        pkl_file (Path): Path to the pickle file
        
    Returns:
        list: List of dictionaries containing article_id and text
    """
    output_rows = []
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            article_id = pkl_file.stem
            
            for text in data:
                output_rows.append({
                    "article_id": article_id,
                    "text": text
                })
        return output_rows
    except Exception as e:
        print(f"Error processing {pkl_file}: {str(e)}")
        return []


def create_dataset_for_inference(output_dir):
    """
    Create dataset from pickle files for inference, returning only article_id and text columns.
    
    Args:
        output_dir (str): Directory containing pickle files
        
    Returns:
        pd.DataFrame: DataFrame with article_id and text columns
    """
    input_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {output_dir}")
    
    test_dataset_path = input_path / "test_df.csv"
    article_ids_to_include = None
    if test_dataset_path.exists():
        test_df = pd.read_csv(test_dataset_path)
        if "article_id" in test_df.columns:
            article_ids_to_include = set(str(a) for a in test_df["article_id"].unique())
    
    pkl_files = list(input_path.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files to process for inference")

    if article_ids_to_include:
        pkl_files = [pkl_file for pkl_file in pkl_files if pkl_file.stem in article_ids_to_include]
    
    output_rows = []
    
    # Process pickle files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all pickle files for processing
        future_to_file = {executor.submit(_process_pkl_file_for_inference_worker, pkl_file): pkl_file for pkl_file in pkl_files}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            pkl_file = future_to_file[future]
            try:
                result = future.result()
                output_rows.extend(result)
                print(f"Processed {pkl_file.name}: {len(result)} rows")
            except Exception as e:
                print(f"Error processing {pkl_file.name}: {str(e)}")
    
    output_df = pd.DataFrame(output_rows, columns=["article_id", "text"])
    print(f"Total rows processed for inference: {len(output_df)}")
    
    return output_df
    



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python data-preprocessing.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    convert_pdfs_to_text(input_directory, output_directory)
    convert_xmls_to_text(input_directory, output_directory)
    decompose_text_to_lines(output_directory)


# Write a function train_test_split, that takes a file path as an argument. Read the file into a pandas dataframe. The header for the csv is 'text,dataset_id,type', The dataset_id has missing values, rebalance the dataframe so that there are equal numbers of rows with missing dataset_id and rows that have dataset_id populated. Then using scikitlearn functions to split the dataframe into two dataframes, train and test, with a split of 80/20. Return both train and test dataframe

def train_test_split(file_path):
    """
    Read a CSV file, rebalance missing dataset_id values, and split into train/test sets.
    
    Args:
        file_path (str): Path to the CSV file with columns 'text,dataset_id,type'
        
    Returns:
        tuple: (train_df, test_df) - Two pandas DataFrames with 80/20 split
    """
    df = pd.read_csv(file_path)
    
    missing_mask = df['dataset_id'].isna() | (df['dataset_id'] == '')
    populated_mask = ~missing_mask
    
    missing_df = df[missing_mask]
    populated_df = df[populated_mask]
    
    min_count = min(len(missing_df), len(populated_df))
    
    if len(missing_df) > min_count:
        missing_df = missing_df.sample(n=min_count, random_state=42)
    elif len(populated_df) > min_count:
        populated_df = populated_df.sample(n=min_count, random_state=42)
    
    balanced_df = pd.concat([missing_df, populated_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, test_df = sklearn_train_test_split(
        balanced_df, 
        test_size=0.2, 
        random_state=42,
        stratify=balanced_df['dataset_id'].isna() | (balanced_df['dataset_id'] == '')
    )
    
    return train_df, test_df
