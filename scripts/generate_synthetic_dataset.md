Your task is implement code in python to generate a dataset csv file. It must perform the following items:

- load the training file (passed as an argument) into a pandas dataframe
- the header for the training file is 'article_id,dataset_id,type'
- remove rows that have 'Missing' in the dataset_id or type columns
- iterate over the unique article_id, dataset_id combination
    - then load the pickle file the article_id file in the directory specified as the second argument
    - the pickle file is an array of strings. Iterate over this array and do:
        - if the dataset_id is found in the current text, then write the text, dataset_id and type to the output_dataset
- use SDV (Synthetic Data Vault) synthetic data generation to create additional synthetic samples
- combine original and synthetic data into final output CSV

The script uses SDV's Gaussian Copula synthesizer to create additional training samples while preserving the statistical properties of the original data.