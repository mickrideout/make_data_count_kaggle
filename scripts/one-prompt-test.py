#!/usr/bin/env python3

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(model_name="microsoft/DialoGPT-medium"):
    """Load the Hugging Face model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def read_article_file(file_path):
    """Read article text from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def run_inference(model, tokenizer, prompt, max_length=512):
    """Run inference on the model with the given prompt."""
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description='Run Hugging Face model inference with article text')
    parser.add_argument('article_file', help='Path to file containing article text')
    parser.add_argument('--model', default='microsoft/DialoGPT-medium', help='Hugging Face model to use')
    
    args = parser.parse_args()
    
    # Hardcoded prompt template with {article_text} placeholder
    prompt_template = """Analyze the following scientific article and identify any dataset citations:

{article_text}

Based on this article, list any datasets that are mentioned or referenced:"""
    
    # Read article text from file
    article_text = read_article_file(args.article_file)
    
    # Replace the placeholder with actual article text
    prompt = prompt_template.format(article_text=article_text)
    
    print("Loading model...")
    model, tokenizer = load_model(args.model)
    
    print("Running inference...")
    response = run_inference(model, tokenizer, prompt)
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(response)


if __name__ == "__main__":
    main()