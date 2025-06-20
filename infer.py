import os
import sys
from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_markdown, decompose_markdown_to_paragraphs
from make_data_count_kaggle.dataset_classification import dummy_classifier
from make_data_count_kaggle.dataset_matching import basic_matching, create_empty_candidate_dataset


def main(input_directory, output_directory):
    # Dataset preprocessing
    convert_pdfs_to_markdown(f'{input_directory}/test', output_directory)
    decompose_markdown_to_paragraphs(output_directory)

    # Candidate generation
    create_empty_candidate_dataset(output_directory)
    basic_matching(output_directory, output_directory)

    # Candidate classification
    dummy_classifier(output_directory, output_directory)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    main(input_directory, output_directory)






