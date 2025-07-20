Write a function in data process that for each of the files in {OUTPUT_DIR}/training_labels.csv (which have the csv header, article_id,dataset_id,type) do the following:
- create a json file {OUTPUT_DIR}/training_labels.json an example csv is:

10.1038_s41396-020-00885-8,IPR000884,Secondary
10.1038_s41396-020-00885-8,IPR001124,Primary
10.1038_s41396-020-00885-8,IPR001577,Secondary

The object is to create a json array of the following entries:

- For each unique article_id in the csv file it will do:
    - create a new json object with the following fields:
        - article_id
        - dataset_mentions: an array of objects
    - for every dataset_id and type pair create a new object with the following fields:
        - dataset_name
        - dataset_type
    - add the new object to the dataset_mentions array
    - add the new json object to the json array


## Sample file contents
[
    {
        "article_id": "10.1038_s41396-020-00885-8",
        "dataset_mentions":
        [
            {
                "dataset_name": "LSMS-ISA",
                "dataset_type": "Primary"
            },
            {
                "dataset_name": "https://doi.org/10.1371/journal.pone.0303785",
                "dataset_type": "Secondary"
            }
        ]
    }
]
