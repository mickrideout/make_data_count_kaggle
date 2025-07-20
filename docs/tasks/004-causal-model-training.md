## Causal model training

- Implement the logic to train a causal model using the huggingface library
- use the model microsoft/Phi-4-mini-instruct
- The training data to iterate over is defined as dataset_dict in train.py
- The prompt should be a template from the file docs/prompts/question-prompt.md . For each article_id in the training data, load the arcitcle text from {OUTPUT_DIR}/{article_id}.txt The text is to be inserted into the prompt where the {article_text} token is. 
- The model output dir should be {OUTPUT_DIR}/causal_model
- Create a new module for this, make_data_count_kaggle/causal_model.py
