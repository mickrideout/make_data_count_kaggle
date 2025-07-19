
In train.py in the data preprocessing phase, create a new function that decomposes the {INPUT_DIR}/train_labels.csv. A sample of the file is in the "## Sample" section below. Based on this file generate two new files in {OUTPUT_DIR}/} one being training_labels.csv and the other testing_labels.csv

For each row in the training_labels.csv file do:
- skip the row if it has Missing in the dataset_id or type columns
- check if the article_id is in the {INPUT_DIR}/train/PDF or {INPUT_DIR}/test/PDF folders (with the .pdf extension added to the article_id). If it exists in the train folder place the row in the train_labels.csv file. If it exists in the test folder place the row in the test_labels.csv file.




## Sample

article_id,dataset_id,type
10.1002_2017jc013030,https://doi.org/10.17882/49388,Primary
10.1002_anie.201916483,Missing,Missing
10.1002_anie.202005531,Missing,Missing
10.1002_anie.202007717,Missing,Missing
10.1002_chem.201902131,Missing,Missing
10.1002_chem.201903120,Missing,Missing
10.1002_chem.202000235,Missing,Missing
10.1002_chem.202001412,Missing,Missing
10.1002_chem.202001668,Missing,Missing
10.1002_chem.202003167,Missing,Missing
10.1002_ece3.3985,Missing,Missing
10.1002_ece3.4466,https://doi.org/10.5061/dryad.r6nq870,Primary
10.1002_ece3.5260,https://doi.org/10.5061/dryad.2f62927,Primary
10.1002_ece3.5395,Missing,Missing
10.1002_ece3.6144,https://doi.org/10.5061/dryad.zw3r22854,Primary
10.1002_ece3.6303,https://doi.org/10.5061/dryad.37pvmcvgb,Primary
10.1002_ece3.6784,Missing,Missing
10.1002_ece3.961,Missing,Missing
10.1002_ece3.9627,https://doi.org/10.5061/dryad.b8gtht7h3,Primary
10.1002_ecs2.1280,https://doi.org/10.5061/dryad.p3fg9,Primar