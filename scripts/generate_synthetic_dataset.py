#!/usr/bin/env python3

import sys
import pandas as pd
import pickle
import os
import random
import re
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.cag import SingleTableProgrammableConstraint

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_dataset.py <training_csv> <pickle_dir> <output_csv>")
        sys.exit(1)
    
    training_csv = sys.argv[1]
    pickle_dir = sys.argv[2]
    output_csv = sys.argv[3]
    
    # Load training file
    df = pd.read_csv(training_csv)
    
    # Remove rows with 'Missing' in dataset_id or type columns
    df_filtered = df[(df['dataset_id'] != 'Missing') & (df['type'] != 'Missing')]
    
    # Get unique article_id, dataset_id combinations
    unique_combinations = df_filtered[['article_id', 'dataset_id', 'type']].drop_duplicates()
    
    # Prepare output data
    output_data = []
    
    # Iterate over unique combinations
    for _, row in unique_combinations.iterrows():
        article_id = row['article_id']
        dataset_id = row['dataset_id']
        data_type = row['type']
        
        # Load pickle file for this article
        pickle_file = os.path.join(pickle_dir, f"{article_id}.pkl")
        
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    text_array = pickle.load(f)
                
                # Iterate over text array
                for text in text_array:
                    if dataset_id in text:
                        output_data.append({
                            'article_id': article_id,
                            'dataset_id': dataset_id,
                            'type': data_type,
                            'text': text
                        })
            except Exception as e:
                print(f"Error loading {pickle_file}: {e}")
        else:
            print(f"Warning: Pickle file not found: {pickle_file}")
    
    # Create output dataframe
    output_df = pd.DataFrame(output_data)
    
    if len(output_df) == 0:
        print("No data found to generate synthetic data from")
        return
    
    print(f"Original data shape: {output_df.shape}")
    
    # Prepare training dataframe for SDV
    training_df = output_df.copy()
    
    # Create metadata for SDV
    print("Creating SDV metadata...")
    metadata = Metadata.detect_from_dataframe(
        data=training_df,
        table_name='dataset_citations',
        infer_sdtypes=True,
        infer_keys='primary_only'
    )
    
    # Create the custom constraint class
    print("Creating SDV constraint: dataset_id must appear in text...")
    
    class DatasetIdInTextConstraint(SingleTableProgrammableConstraint):
        """The DatasetIdInTextConstraint enforces that dataset_id appears in the text"""
        
        def transform(self, data):
            """Preprocess the data - no transformation needed"""
            return data
        
        def get_updated_metadata(self, metadata):
            """No metadata changes needed"""
            return metadata
        
        def reverse_transform(self, synthetic_data):
            """No reverse transformation needed"""
            return synthetic_data
        
        def is_valid(self, data):
            """Check that dataset_id appears in text for every row"""
            return data.apply(lambda row: row['dataset_id'] in row['text'], axis=1)
    
    # Create the constraint instance
    dataset_constraint = DatasetIdInTextConstraint()
    
    # Initialize SDV synthesizer
    print("Initializing SDV CTGAN synthesizer...")
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=1000,
        verbose=True,
        cuda=True
    )
    
    # Add constraints to the synthesizer
    print("Adding constraints to synthesizer...")
    synthesizer.add_constraints(constraints=[dataset_constraint])
    
    # Train SDV model
    print("Training SDV model...")
    synthesizer.fit(training_df)
    
    # Generate synthetic data
    target_rows = 1000
    synthetic_rows_needed = target_rows - len(output_df)
    
    print(f"Generating {synthetic_rows_needed} synthetic rows...")
    synthetic_df = synthesizer.sample(num_rows=synthetic_rows_needed)
    
    print(f"Successfully generated {len(synthetic_df)} synthetic rows with constraint enforcement")
    
    # Combine original and synthetic data
    final_df = pd.concat([output_df, synthetic_df], ignore_index=True)
    
    # Save to output file
    final_df.to_csv(output_csv, index=False)
    
    print(f"Generated dataset saved to {output_csv}")
    print(f"Total rows: {len(final_df)}")
    print(f"Original rows: {len(output_df)}")
    print(f"Synthetic rows: {len(final_df) - len(output_df)}")

if __name__ == "__main__":
    main()