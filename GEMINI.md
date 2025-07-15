# Gemini Project: Make Data Count Kaggle Competition

This project aims to solve the "Make Data Count" Kaggle competition. The primary goal is to develop a model that can accurately identify and classify data citations within the full text of scientific articles.

## Overview

The model must perform two main tasks:
1.  **Identify Data Citations:** Locate all references to research datasets within a given scientific paper.
2.  **Classify Citation Type:** For each identified citation, classify it as either:
    *   **Primary:** Data generated specifically for the study described in the paper.
    *   **Secondary:** Data reused from existing sources.

The ultimate goal is to create a highly performant model that can be used to continuously update the Make Data Count (MDC) Data Citation Corpus with new, contextualized links between papers and datasets.

## Project Structure

The project is organized as follows:

-   `train.py`: Script for training the model.
-   `infer.py`: Script for running inference on new data to generate predictions.
-   `requirements.txt`: A list of Python dependencies for this project.
-   `make_data_count_kaggle/`: Source code for the project.
    -   `data_preprocessing.py`: Handles the cleaning and preparation of the text data.
    -   `dataset_classification.py`: Contains the data loading and feature engineering logic for the classification task.
    -   `dataset_matching.py`: Contains the data loading and feature engineering logic for the matching task.

## How to Run

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training

To train the model, run the `train.py` script:

```bash
python train.py
```

### 3. Inference

To generate a submission file, run the `infer.py` script on the test data:

```bash
python infer.py
```

This will produce a `submission.csv` file in the format required by the Kaggle competition.
