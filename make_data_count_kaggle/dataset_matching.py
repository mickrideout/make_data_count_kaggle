import glob
import os
from tqdm import tqdm
import re
import pandas as pd

CANDIDATE_CSV_FILE = "candidate_dataset.csv"
CANDIDATE_HEADER = ["article_id", "candidate_matcher", "dataset_id", "candidate_text"]

def create_empty_candidate_dataset(output_dir):
    """
    Create an empty candidate dataset.
    """
    with open(f"{output_dir}/{CANDIDATE_CSV_FILE}", "w") as f:
        f.write(",".join(CANDIDATE_HEADER) + "\n")

def basic_matching(input_dir, output_dir, context_paragraphs=2):
    """
    Basic matching of the dataset. Adapted from https://www.kaggle.com/code/srinivasta/make-data-count-finding-data-references
    """

    MATCHER_NAME = "basic_matching"

    print(f"Performing {MATCHER_NAME} matching")

    def normalize_doi(doi):
        doi = doi.strip().lower()
        if doi.endswith(')') and doi.count('(') < doi.count(')'):
            doi = doi[:-1]
        if doi.startswith("https://doi.org/"):
            return doi
        if doi.startswith("doi:"):
            doi = doi[4:]
        if doi.startswith("10."):
            return "https://doi.org/" + doi
        return doi

    def extract_references(text):
        dois = re.findall(r'\b10\.\d{4,9}/[-._;()/:a-z0-9]+', text, flags=re.I)
        accessions = re.findall(r'\b(GSE\d+|E-[A-Z]+-\d+|PRJ[EDNA]\d+|PDB\s*\w+)\b', text, flags=re.I)
        references = []

        for doi in dois:
            doi_full = normalize_doi(doi)
            references.append(doi_full)

        for acc in accessions:
            acc_clean = acc.replace(" ", "").upper()
            references.append(acc_clean)

        return references
    
    df = pd.DataFrame(columns=CANDIDATE_HEADER)

    for file in tqdm(glob.glob(f'{output_dir}/*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            article_id = os.path.basename(file).rsplit(".", 1)[0]
            print(f"Processing {article_id}")
            
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    match_list = extract_references(paragraph)
                    if not match_list:
                        continue
                    
                    start_index = max(0, i - context_paragraphs)
                    end_index = min(len(paragraphs), i + context_paragraphs + 1)
                    context_text = "\n\n".join(paragraphs[start_index:end_index])

                    for match in match_list:
                        df.loc[len(df)] = [article_id, MATCHER_NAME, match, context_text]

    df.to_csv(f"{output_dir}/{CANDIDATE_CSV_FILE}", index=False)
