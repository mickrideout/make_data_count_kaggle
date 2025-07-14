import os
import glob
import pickle
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from mrkdwn_analysis import MarkdownAnalyzer


def convert_pdfs_to_markdown(input_dir, output_dir):
    """
    Convert all PDF files in input_dir to markdown files in output_dir using marker.
    
    Args:
        input_dir (str): Directory containing PDF files to convert
        output_dir (str): Directory where markdown files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if output_path.exists() and output_path.is_dir() and len(list(output_path.iterdir())) > 0:
        print(f"Output directory already exists and contains files: {output_dir}")
        return

    
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
            if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive' or 'KAGGLE_CONTAINER_NAME' in os.environ:
                config = {
                    "output_format": "markdown",
                    "disable_image_extraction": True,
                    "kaggle_mode": True,
                }
            else:
                config = {
                    "output_format": "markdown",
                    "disable_image_extraction": True,
                }
            config_parser = ConfigParser(config)

            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service()
            )
            markdown_content = converter(pdf_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text_from_rendered(markdown_content)[0])
                
            print(f"Successfully converted {pdf_path.name}")
            
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {str(e)}")
    
    print(f"Conversion complete. Markdown files saved to {output_dir}")


def decompose_markdown_to_paragraphs(output_dir):
    """
    Decompose all markdown files in output_dir into paragraphs using markdown-analysis
    and save the paragraph arrays as pickle files.
    
    Args:
        output_dir (str): Directory containing markdown files to decompose
    """
    output_path = Path(output_dir)

    if output_path.exists() and output_path.is_dir() and len(list(output_path.iterdir())) > 0:
        print(f"Output directory already exists and contains files: {output_dir}")
        return
    
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

