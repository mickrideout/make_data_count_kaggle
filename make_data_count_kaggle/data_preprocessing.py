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
from datasets import Dataset, DatasetDict
from langchain.text_splitter import RecursiveCharacterTextSplitter


def remove_references_section(text):
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()
        
        obvious_patterns = [
            r'^REFERENCES?$',
            r'^\d+\.?\s+REFERENCES?$',
            r'^\d+\.?\s+References?$',
            r'^References?:?$',
            r'^BIBLIOGRAPHY$',
            r'^\d+\.?\s+BIBLIOGRAPHY$',
            r'^\d+\.?\s+Bibliography$',
            r'^Bibliography:?$',
            r'^Literature\s+Cited$',
            r'^Works\s+Cited$'
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in obvious_patterns):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i+1:i+4]
            has_citations = False
            
            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or    # (2020)
                        re.search(r'\d{4}\.', follow_line) or       # 2020.
                        'doi:' in follow_line.lower() or           # DOI
                        ' et al' in follow_line.lower()):          # et al
                        has_citations = True
                        break
            
            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    
    return text.strip()

def clean_text(text):
    """
    Clean up text by removing references section.
    
    Args:
        text (str): Raw text from PyMuPDF
        
    Returns:
        str: Cleaned text with references section removed
    """
    # Remove references section
    cleaned_text = remove_references_section(text)
    
    return cleaned_text


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
    Worker function to decompose a single text file into chunks using RecursiveCharacterTextSplitter.
    """
    text_path = Path(text_file)
    output_path = Path(output_dir)
    filename = text_path.stem
    pickle_file = output_path / f"{filename}.pkl"

    if pickle_file.exists():
        return f"Skipping {text_path.name} - {pickle_file.name} already exists"

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use RecursiveCharacterTextSplitter to split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_text(content)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunks, f)
            
        return f"Successfully decomposed {text_path.name} into {len(chunks)} chunks"
    except Exception as e:
        return f"Error decomposing {text_path.name}: {str(e)}"


def decompose_text_to_chunks(output_dir):
    """
    Decompose all text in output_dir into chunks using RecursiveCharacterTextSplitter 
    and save the chunk arrays as pickle files in parallel.
    
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


def convert_labels_csv_to_json(output_dir):
    """
    Convert training_labels.csv to JSON format with dataset_mentions structure.
    Creates a JSON array where each article has an array of dataset_mentions objects.
    
    Args:
        output_dir (str): Directory containing training_labels.csv and where training_labels.json will be saved
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Process training labels
    training_csv = output_path / "training_labels.csv"
    if not training_csv.exists():
        raise ValueError(f"training_labels.csv not found in {output_dir}")
    
    df = pd.read_csv(training_csv)
    
    # Group by article_id to create the JSON structure
    json_data = []
    
    for article_id in df['article_id'].unique():
        # Get all datasets for this article
        article_datasets = df[df['article_id'] == article_id]
        
        # Create dataset_mentions array
        dataset_mentions = []
        for _, row in article_datasets.iterrows():
            dataset_mentions.append({
                "dataset_name": row['dataset_id'],
                "dataset_type": row['type']
            })
        
        # Create article object
        article_obj = {
            "article_id": article_id,
            "dataset_mentions": dataset_mentions
        }
        
        json_data.append(article_obj)
    
    # Save to JSON
    training_json = output_path / "training_labels.json"
    with open(training_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created training_labels.json with {len(json_data)} articles")


def create_huggingface_dataset(output_dir, train_ratio=0.8):
    """
    Convert training_labels.json to a HuggingFace Dataset with train/test split.
    
    Args:
        output_dir (str): Directory containing training_labels.json
        train_ratio (float): Ratio for train split (default 0.8 for 80/20 split)
        
    Returns:
        DatasetDict: HuggingFace dataset with 'train' and 'test' splits
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Load the training labels JSON
    training_json = output_path / "training_labels.json"
    if not training_json.exists():
        raise ValueError(f"training_labels.json not found in {output_dir}")
    
    with open(training_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} articles from training_labels.json")
    
    # Convert to HuggingFace Dataset format
    # Flatten the data structure - create one row per (article_id, dataset_id, type) combination
    flattened_data = []
    for article in data:
        article_id = article['article_id']
        dataset_mentions = article['dataset_mentions']
        
        # Create one row for each dataset in the article
        for mention in dataset_mentions:
            flattened_data.append({
                'article_id': article_id,
                'dataset_id': mention['dataset_name'],
                'type': mention['dataset_type']
            })
    
    print(f"Flattened to {len(flattened_data)} dataset citations")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(flattened_data)
    
    # Create train/test split
    train_test_split = dataset.train_test_split(train_size=train_ratio, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    
    print(f"Created dataset with {len(dataset_dict['train'])} training examples and {len(dataset_dict['test'])} test examples")
    
    return dataset_dict


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python data-preprocessing.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    convert_pdfs_to_text(input_directory, output_directory)
    convert_xmls_to_text(input_directory, output_directory)
    decompose_text_to_chunks(output_directory)
