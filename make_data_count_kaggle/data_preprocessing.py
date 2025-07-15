import os
import glob
import pickle
import pymupdf4llm
import re
from pathlib import Path
from mrkdwn_analysis import MarkdownAnalyzer


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


def convert_pdfs_to_markdown(input_dir, output_dir):
    """
    Convert all PDF files in input_dir to markdown files in output_dir using pymupdf4llm.
    
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
    
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file)
        filename = pdf_path.stem
        
        output_file = output_path / f"{filename}.md"
        
        print(f"Converting {pdf_path.name} to {output_file.name}")
        
        try:
            md_text = pymupdf4llm.to_markdown(pdf_file)
            
            if md_text.strip():
                # Clean up column breaks and create natural paragraphs
                cleaned_md = clean_markdown_text(md_text)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_md)
                    
                print(f"Successfully converted {pdf_path.name}")
            else:
                print(f"No text content found in {pdf_path.name}")
                
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {str(e)}")
    
    print(f"Conversion complete. Markdown files saved to {output_dir}")


def decompose_text_to_paragraphs(output_dir):
    """
    Decompose all text in output_dir into paragraphs using markdown-analysis
    and save the paragraph arrays as pickle files.
    
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
    
    for md_file in md_files:
        md_path = Path(md_file)
        filename = md_path.stem
        
        pickle_file = output_path / f"{filename}.pkl"
        
        print(f"Decomposing {md_path.name} into paragraphs")
        
        try:
              
            analyzer = MarkdownAnalyzer(md_file)
            paragraphs = analyzer.identify_paragraphs()['Paragraph']
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(paragraphs, f)
                
            print(f"Successfully decomposed {md_path.name} into {len(paragraphs)} paragraphs")
            
        except Exception as e:
            print(f"Error decomposing {md_path.name}: {str(e)}")
    
    print(f"Decomposition complete. Pickle files saved to {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python data-preprocessing.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    convert_pdfs_to_markdown(input_directory, output_directory)

