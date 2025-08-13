import os
import sys
import pandas as pd
import json
import re
import string

# Set PyTorch CUDA memory management environment variables for better memory handling
# Prevent large allocations and improve memory fragmentation handling
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'  # Disable memory caching for more predictable behavior

from make_data_count_kaggle.data_preprocessing import convert_pdfs_to_text, convert_xmls_to_text, decompose_text_to_lines, decompose_train_labels, create_dataset_for_training, create_dataset_for_inference, train_test_split
from make_data_count_kaggle.evaluation import calculate_f1_score



def parser(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([element for element in item])
            else:
                item = item
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []

def get_conv_template(name):
    if name == "ie_as_qa":
        return Conversation(
            name="ie_as_qa",
            system="A virtual assistant answers questions from a user based on the provided text.",
            roles=("USER", "ASSISTANT"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    else:
        raise ValueError(f"Unknown conversation template: {name}")

class SeparatorStyle:
    ADD_COLON_TWO = "add_colon_two"

class Conversation:
    def __init__(self, name, system, roles, messages, offset, sep_style, sep, sep2):
        self.name = name
        self.system = system
        self.roles = roles
        self.messages = list(messages)
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
    
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

def preprocess_instance(source):    
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

def get_response(responses):
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses

def inference(model, examples, max_new_tokens=256):
    from vllm import SamplingParams
    
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])
    responses = model.generate(prompts, sampling_params)
    responses_correct_order = []
    response_set = {response.prompt: response for response in responses}
    for prompt in prompts:
        assert prompt in response_set
        responses_correct_order.append(response_set[prompt])
    responses = responses_correct_order
    outputs = get_response([output.outputs[0].text for output in responses])
    return outputs

def dataset_classification_examples(dataset_df):
    examples = []
    
    for _, row in dataset_df.iterrows():
        # Get the article text
        text_content = row['text']
        dataset_id = row['dataset_id']
        
        example = {
            "conversations": [
                {"from": "human", "value": f"{text_content}"},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": f"The dataset mentioned is {dataset_id}. Is it a primary or secondary source?"},
                {f"from": "gpt", "value": f"[{dataset_id}]"}
            ]
        }
        
        examples.append({
            'article_id': row['article_id'],
            'text': text_content,
            'example': example
        })
    
    return examples

def create_ner_examples_from_dataset(dataset_df):
    examples = []
    
    for _, row in dataset_df.iterrows():
        # Get the article text
        text_content = row['text']
        
        # Create NER example for dataset identification
        # We use a generic entity type "dataset" to identify dataset mentions
        example = {
            "conversations": [
                {"from": "human", "value": f"{text_content}"},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": "What describes the datasets that are mentioned in this text?"},
            ]
        }
        
        examples.append({
            'article_id': row['article_id'],
            'text': text_content,
            'example': example
        })
    
    return examples

def run_uniner_inference(dataset_df, output_directory, model_path, tensor_parallel_size=1, batch_size=100):
    from vllm import LLM
    import torch
    import gc
    
    print(f"Loading UniversalNER model from {model_path}...")
    
    # Load the UniversalNER model
    import os
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", None) is not None:
        print("Running in a Kaggle environment.")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.8,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            disable_log_stats=True,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=2048,
            swap_space=0,
            disable_custom_all_reduce=True,
        )
    else:
        print("Running in a local environment.")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            disable_log_stats=True,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=2048,
            swap_space=0,
            disable_custom_all_reduce=True,

        )

    # STAGE 1: Dataset identification
    print("STAGE 1: Identifying datasets...")
    
    # Create NER examples from the dataset
    ner_examples = create_ner_examples_from_dataset(dataset_df)

    
    print(f"Processing {len(ner_examples)} articles with UniversalNER...")
    
    # Run inference in batches to manage memory
    stage1_results = []
    
    for i in range(0, len(ner_examples), batch_size):
        batch = ner_examples[i:i + batch_size]
        batch_examples = [item['example'] for item in batch]
        
        # Run inference
        outputs = inference(llm, batch_examples, max_new_tokens=512)
        
        # Process outputs and associate with article IDs
        for j, output in enumerate(outputs):
            article_id = batch[j]['article_id']
            parsed_entities = parser(output)
            
            stage1_results.append({
                'article_id': article_id,
                'entities': parsed_entities,
                'raw_output': output,
                'text': batch[j]['text']
            })
        
        # Clear memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Processed batch {i//batch_size + 1}/{(len(ner_examples) + batch_size - 1)//batch_size}")

    
    # STAGE 2: Classification of identified datasets
    print("STAGE 2: Classifying datasets as primary or secondary...")
    
    # Create dataset for classification stage
    classification_data = []
    for result in stage1_results:
        if result['entities']:  # Only process articles with identified datasets
            for entity in result['entities']:
                if isinstance(entity, (tuple, list)) and len(entity) >= 1:
                    dataset_id = str(entity[0])
                else:
                    dataset_id = str(entity)
                
                classification_data.append({
                    'article_id': result['article_id'],
                    'text': result['text'],
                    'dataset_id': dataset_id
                })
    
    if classification_data:
        classification_df = pd.DataFrame(classification_data)
        
        # Create classification examples
        classification_examples = dataset_classification_examples(classification_df)
        
        print(f"Classifying {len(classification_examples)} dataset mentions...")
        
        stage2_results = []
        
        for i in range(0, len(classification_examples), batch_size):
            batch = classification_examples[i:i + batch_size]
            batch_examples = [item['example'] for item in batch]
            
            # Run classification inference
            outputs = inference(llm, batch_examples, max_new_tokens=256)
            
            # Process classification outputs
            for j, output in enumerate(outputs):
                article_id = batch[j]['article_id']
                text_content = batch[j]['text']
                
                # Parse classification result
                classification = output.strip().lower()
                if 'secondary' in classification:
                    dataset_type = 'Secondary'
                else:
                    dataset_type = 'Primary'  # Default to Primary
                
                stage2_results.append({
                    'article_id': article_id,
                    'dataset_id': classification_df.iloc[i + j]['dataset_id'],
                    'type': dataset_type,
                    'text': text_content,
                    'classification_output': output
                })
            
            # Clear memory periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Classified batch {i//batch_size + 1}/{(len(classification_examples) + batch_size - 1)//batch_size}")
        
        return stage2_results
    else:
        print("No datasets identified in stage 1, returning empty results")
        return []

def main(input_directory, output_directory, model_path, tensor_parallel_size=1, batch_size=100):
    print("Starting UniversalNER inference pipeline with memory optimizations...")
    
    # Import torch for memory management
    import torch
    import gc
    
    # Dataset preprocessing for inference
    convert_xmls_to_text(f"{input_directory}/test", output_directory)
    convert_pdfs_to_text(f"{input_directory}/test", output_directory)
    decompose_text_to_lines(output_directory)
    dataset_df = create_dataset_for_inference(output_directory)
    
    # Clear any Python garbage before model loading
    gc.collect()
    
    # Additional memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Check available memory before proceeding
        memory_info = torch.cuda.mem_get_info()
        available_memory = memory_info[0] / 1024**3
        total_memory = memory_info[1] / 1024**3
        print(f"Pre-inference memory check: {available_memory:.2f}/{total_memory:.2f} GB available")
        
        if available_memory < 5.0:
            print(f"WARNING: Low GPU memory ({available_memory:.2f} GB). Consider using smaller model or batch size.")

    # UniversalNER inference
    SUBMISSION_FILE = "submission.csv"

    try:
        print("Running UniversalNER inference...")
        inference_results = run_uniner_inference(dataset_df, output_directory, model_path, tensor_parallel_size, batch_size)
    except Exception as e:
        print(f"UniversalNER inference failed: {e}")
        # Create empty results as fallback
        inference_results = []

    # Convert inference results to output dataframe
    output_data = []
    row_id = 0
    
    # Process results from two-stage inference
    for result in inference_results:
        output_data.append({
            'article_id': result['article_id'],
            'dataset_id': result['dataset_id'],
            'type': result['type'],
        })
        row_id += 1
            
    # Create DataFrame with proper columns even if empty
    if output_data:
        output_df = pd.DataFrame(output_data)
    else:
        output_df = pd.DataFrame(columns=['article_id', 'dataset_id', 'type'])

    # Save submission file
    output_df.to_csv(f"{output_directory}/{SUBMISSION_FILE}", index=False, escapechar='\\', quoting=1)
    print(f"Saved submission file: {output_directory}/{SUBMISSION_FILE}")
    print(f"Generated {len(output_data)} predictions")
    
    # F1-score evaluation step (only if train_labels.csv exists)
    train_labels_path = f"{input_directory}/train_labels.csv"
    if os.path.exists(train_labels_path):
        try:
            print("\nEvaluating F1-score against ground truth...")
            
            # Load ground truth labels
            ground_truth_df = pd.read_csv(train_labels_path)
            
            # Filter submission to only include article_ids that exist in ground truth
            submission_df = pd.read_csv(f"{output_directory}/{SUBMISSION_FILE}")
            
            # Get common article_ids between ground truth and submission
            common_article_ids = set(ground_truth_df['article_id']).intersection(set(submission_df['article_id']))
            
            if common_article_ids:
                # Filter both dataframes to only include common article_ids
                ground_truth_filtered = ground_truth_df[ground_truth_df['article_id'].isin(common_article_ids)]
                submission_filtered = submission_df[submission_df['article_id'].isin(common_article_ids)]
                
                # Calculate F1 score
                f1, precision, recall, tp, fp, fn = calculate_f1_score(
                    ground_truth_filtered, 
                    submission_filtered[['article_id', 'dataset_id', 'type']], 
                    output_directory
                )
                
                print(f"F1 Score: {f1:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"True Positives: {tp}")
                print(f"False Positives: {fp}")
                print(f"False Negatives: {fn}")
                print(f"Evaluated on {len(common_article_ids)} common articles")
            else:
                print("No common article_ids found between ground truth and predictions")
                
        except Exception as e:
            print(f"Error during F1-score evaluation: {e}")
    else:
        print(f"No train_labels.csv found at {train_labels_path}, skipping evaluation")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python infer_uniner.py <input_dir> <output_dir> <model_path> <tensor_parallel_size> <batch_size>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_path = sys.argv[3]
    tensor_parallel_size = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    main(input_directory, output_directory, model_path, tensor_parallel_size, batch_size)