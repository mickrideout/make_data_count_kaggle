import os
import glob
import pickle
import signal
import pymupdf4llm
import re
from pathlib import Path
import concurrent.futures
from functools import partial


def clean_markdown_text(md_text):
    """
    Clean up markdown text by removing column breaks and creating natural paragraphs.
    
    Args:
        md_text (str): Raw markdown text from pymupdf4llm
        
    Returns:
        str: Cleaned markdown text with natural paragraphs
    """
    # First, normalize line endings
    md_text = md_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into lines and process
    lines = md_text.split('\n')
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
    
    # Remove multiple consecutive spaces
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # Remove multiple consecutive newlines (keep max 2)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Remove trailing spaces from lines
    cleaned_text = re.sub(r' +$', '', cleaned_text, flags=re.MULTILINE)
    
    return cleaned_text


def _convert_pdf_to_markdown_worker(pdf_file, output_dir):
    """
    Worker function to convert a single PDF to markdown with a 10-minute timeout.
    """
    pdf_path = Path(pdf_file)
    output_path = Path(output_dir)
    filename = pdf_path.stem
    output_file = output_path / f"{filename}.md"

    if output_file.exists():
        return f"Skipping {pdf_path.name} - {output_file.name} already exists"

    # Define a handler for the timeout
    def handler(signum, frame):
        raise TimeoutError("PDF conversion timed out after 10 minutes")

    # Set the signal handler and a 10-minute (600 seconds) alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600)

    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        if md_text.strip():
            cleaned_md = clean_markdown_text(md_text)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_md)
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


def convert_pdfs_to_markdown(input_dir, output_dir):
    """
    Convert all PDF files in input_dir to markdown files in output_dir using pymupdf4llm in parallel.
    
    Args:
        input_dir (str): Directory containing PDF files to convert
        output_dir (str): Directory where markdown files will be saved
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
        convert_func = partial(_convert_pdf_to_markdown_worker, output_dir=output_dir)
        
        # Process files in parallel
        results = executor.map(convert_func, pdf_files)
        
        # Output results
        for result in results:
            print(result)
            
    print(f"Conversion complete. Markdown files saved to {output_dir}")


def _decompose_text_worker(md_file, output_dir):
    """
    Worker function to decompose a single markdown file into paragraphs.
    """
    md_path = Path(md_file)
    output_path = Path(output_dir)
    filename = md_path.stem
    pickle_file = output_path / f"{filename}.pkl"

    if pickle_file.exists():
        return f"Skipping {md_path.name} - {pickle_file.name} already exists"

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Paragraphs are separated by blank lines
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(paragraphs, f)
            
        return f"Successfully decomposed {md_path.name} into {len(paragraphs)} paragraphs"
    except Exception as e:
        return f"Error decomposing {md_path.name}: {str(e)}"


def decompose_text_to_paragraphs(output_dir):
    """
    Decompose all text in output_dir into paragraphs and save the paragraph 
    arrays as pickle files in parallel.
    
    Args:
        output_dir (str): Directory containing markdown files to decompose
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    md_files = glob.glob(str(output_path / "*.md"), recursive=False)
    
    if not md_files:
        print(f"No markdown files found in {output_dir}")
        return
    
    print(f"Found {len(md_files)} markdown files to decompose")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function to pass the output_dir to the worker
        decompose_func = partial(_decompose_text_worker, output_dir=output_dir)
        
        # Process files in parallel
        results = executor.map(decompose_func, md_files)
        
        # Output results
        for result in results:
            print(result)
            
    print(f"Decomposition complete. Pickle files saved to {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python data-preprocessing.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    convert_pdfs_to_markdown(input_directory, output_directory)
    decompose_text_to_paragraphs(output_directory)