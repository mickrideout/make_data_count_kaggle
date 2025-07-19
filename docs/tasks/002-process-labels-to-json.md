Write a function in data process that for each of the files in {OUTPUT_DIR}/training_labels.csv and {OUTPUT_DIR}/testing_labels.csv (which have the csv header, article_id,dataset_id,type) do the following:
- create a json file {OUTPUT_DIR}/training_labels.json or testing_labels.json which groups article_ids together.. an example csv is:

10.1038_s41396-020-00885-8,IPR000884,Secondary
10.1038_s41396-020-00885-8,IPR001124,Primary
10.1038_s41396-020-00885-8,IPR001577,Secondary


produces the following json format:
[{
    "article_id": "10.1038_s41396-020-00885-8",
    "dataset_ids": ["IPR000884", "IPR001124", "IPR001577"],
    "types": ["Secondary", "Primary", "Secondary"]
}, {
}]