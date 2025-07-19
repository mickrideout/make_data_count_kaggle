import os
import glob
import pickle
import signal
import pymupdf4llm
import xml.etree.ElementTree as ET
import re
import pandas as pd
from pathlib import Path
import concurrent.futures
from functools import partial
import json


def clean_text(text):
    """
    Clean up text by removing column breaks and creating natural paragraphs.
    
    Args:
        text (str): Raw text from pymupdf4llm
        
    Returns:
        str: Cleaned text with natural paragraphs
    """
    # First, normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into lines and process
    lines = text.split('\n')
    cleaned_lines = []
    current_paragraph = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            # If we have accumulated text, join it and add to cleaned lines
            if current_paragraph:
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append('')
            continue
        
        # Check if this line should start a new paragraph
        should_start_new = (
            line.startswith('#') or  # Headers
            line.startswith('- ') or  # List items
            line.startswith('* ') or  # List items
            line.startswith('1. ') or  # Numbered lists
            line.startswith('**') or   # Bold text
            line.startswith('```') or  # Code blocks
            line.startswith('|') or    # Table rows
            (i > 0 and lines[i-1].strip() and 
             lines[i-1].strip().endswith(('.', '!', '?', ':', ';')))  # Previous line ended with punctuation
        )
        
        # If we should start a new paragraph and we have accumulated text
        if should_start_new and current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
            current_paragraph = []
        
        # Add current line to paragraph or start new one
        if should_start_new:
            cleaned_lines.append(line)
        else:
            current_paragraph.append(line)
    
    # Add any remaining paragraph
    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))
    
    # Join all lines
    cleaned_text = '\n'.join(cleaned_lines) 
    
    return cleaned_text


def _convert_pdf_to_text_worker(pdf_file, output_dir):
    """
    Worker function to convert a single PDF to text with a 2-minute timeout.
    """
    pdf_path = Path(pdf_file)
    output_path = Path(output_dir)
    filename = pdf_path.stem
    output_file = output_path / f"{filename}.txt"

    if output_file.exists():
        return f"Skipping {pdf_path.name} - {output_file.name} already exists"

    # Define a handler for the timeout
    def handler(signum, frame):
        raise TimeoutError("PDF conversion timed out after 10 minutes")

    # Set the signal handler and a 2-minute (120 seconds) alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(120)

    try:
        text = pymupdf4llm.to_markdown(str(pdf_path))
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
    Convert all PDF files in input_dir to text files in output_dir using pymupdf4llm in parallel.
    
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
        text = "".join(root.itertext())
        
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
    Worker function to decompose a single text file into paragraphs.
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
        
        # Paragraphs are separated by blank lines
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(paragraphs, f)
            
        return f"Successfully decomposed {text_path.name} into {len(paragraphs)} paragraphs"
    except Exception as e:
        return f"Error decomposing {text_path.name}: {str(e)}"


def decompose_text_to_paragraphs(output_dir):
    """
    Decompose all text in output_dir into paragraphs and save the paragraph 
    arrays as pickle files in parallel.
    
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
    Convert training_labels.csv and testing_labels.csv to JSON format.
    Groups dataset_ids and types by article_id.
    
    Args:
        output_dir (str): Directory containing the CSV files and where JSON files will be saved
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Process training labels
    training_csv = output_path / "training_labels.csv"
    if training_csv.exists():
        df = pd.read_csv(training_csv)
        
        # Group by article_id
        grouped = df.groupby('article_id').agg({
            'dataset_id': list,
            'type': list
        }).reset_index()
        
        # Convert to list of dictionaries
        json_data = []
        for _, row in grouped.iterrows():
            json_data.append({
                'article_id': row['article_id'],
                'dataset_ids': row['dataset_id'],
                'types': row['type']
            })
        
        # Save to JSON
        training_json = output_path / "training_labels.json"
        with open(training_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created training_labels.json with {len(json_data)} articles")
    else:
        print("training_labels.csv not found")
    
    # Process testing labels
    testing_csv = output_path / "testing_labels.csv"
    if testing_csv.exists():
        df = pd.read_csv(testing_csv)
        
        # Group by article_id
        grouped = df.groupby('article_id').agg({
            'dataset_id': list,
            'type': list
        }).reset_index()
        
        # Convert to list of dictionaries
        json_data = []
        for _, row in grouped.iterrows():
            json_data.append({
                'article_id': row['article_id'],
                'dataset_ids': row['dataset_id'],
                'types': row['type']
            })
        
        # Save to JSON
        testing_json = output_path / "testing_labels.json"
        with open(testing_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created testing_labels.json with {len(json_data)} articles")
    else:
        print("testing_labels.csv not found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python data-preprocessing.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    convert_pdfs_to_text(input_directory, output_directory)
    convert_xmls_to_text(input_directory, output_directory)
    decompose_text_to_paragraphs(output_directory)
