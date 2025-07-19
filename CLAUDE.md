# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for "Make Data Count" that identifies and classifies data citations within scientific articles. The goal is to build a model that can:

1. **Identify Data Citations**: Locate all references to research datasets within scientific papers
2. **Classify Citation Type**: Categorize each citation as either "Primary" (data generated for the study) or "Secondary" (reused data)

The project processes scientific articles in PDF and XML formats, extracts text, and uses pattern matching to identify dataset references like DOIs and accession numbers.

## Code Architecture

The codebase follows a modular pipeline architecture with three main processing stages:

### Core Modules

- **`data_preprocessing.py`**: Converts PDFs/XMLs to text and decomposes into paragraphs
  - Uses `pymupdf4llm` for PDF conversion with 2-minute timeouts per file
  - Parallel processing with `ProcessPoolExecutor`
  - Outputs `.txt` files and pickled paragraph arrays

- **`dataset_matching.py`**: Candidate generation using pattern matching
  - Extracts DOIs (`10.xxxx/...`) and accession numbers (`GSE`, `E-`, `PRJ`, `PDB`)
  - Creates candidate dataset CSV with context paragraphs
  - Based on regex patterns for common dataset identifiers

- **`dataset_classification.py`**: Classification of candidates (currently dummy implementation)
  - Takes candidate CSV and outputs submission format
  - Currently classifies all candidates as "Primary"

- **`evaluation.py`**: F1 score calculation and performance metrics
  - Micro F1 score based on (article_id, dataset_id, type) tuples
  - Outputs TP/FP/FN analysis files

### Pipeline Flow

1. **Preprocessing**: `convert_xmls_to_text()` → `convert_pdfs_to_text()` → `decompose_text_to_paragraphs()`
2. **Candidate Generation**: `create_empty_candidate_dataset()` → `basic_matching()`
3. **Classification**: `dummy_classifier()`
4. **Evaluation**: `calculate_f1_score()` (training only)

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training Pipeline
```bash
python train.py <input_dir> <output_dir>
# Example: python train.py /path/to/provided/data /path/to/output
```

### Inference Pipeline
```bash
python infer.py <input_dir> <output_dir>
# Example: python infer.py /path/to/test/data /path/to/output
```

### Individual Module Testing
```bash
# Data preprocessing only
python make_data_count_kaggle/data_preprocessing.py <input_dir> <output_dir>
```

## Code Conventions

- Only add code comments to function definitions
- Always use British spelling
- Functions use parallel processing where applicable
- MLflow integration for experiment tracking (training only)

## Key Dependencies

- `PyMuPDF` / `pymupdf4llm`: PDF text extraction
- `pandas`: Data manipulation
- `mlflow`: Experiment tracking
- `fuzzywuzzy`: Text matching utilities
- `xml.etree.ElementTree`: XML parsing

## File Formats

- **Input**: PDF and XML scientific articles
- **Intermediate**: `.txt` files, `.pkl` paragraph arrays, `candidate_dataset.csv`
- **Output**: `submission.csv` with columns: `row_id`, `article_id`, `dataset_id`, `type`