# src/utils/utils.py

import json
import pandas as pd
import logging

import torch
from src.utils.config import CONFIG

# Configure logger for the utils module
logger = logging.getLogger(__name__)

def count_json_lines(file_path):
    """
    Count the number of lines in a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        int: Number of lines in the file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

def load_first_line_from_json(file_path):
    """
    Load and parse the first line from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict: The parsed JSON object from the first line of the file.
    
    Raises:
        IOError: If there's an issue reading the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {e}")
        raise IOError(f"Error reading from {file_path}: {e}")

def preprocess_data(row):
    """
    Preprocess a single row of data to construct the input and output sequences for the model.
    
    Args:
        row (pd.Series): A pandas Series representing a single row of data.
    
    Returns:
        pd.Series: A pandas Series containing the processed input and output sequences.
    """
    print(f"\n--- preprocess_data {row['story_id']} ---")
    try:
        # Extract fields from the row
        premise = row.get('premise', "Missing premise")
        initial = row.get('initial', "Missing initial")
        original_ending = row.get('original_ending', "Missing original_ending")
        counterfactual = row.get('counterfactual', "Missing counterfactual")
        edited_endings = row.get('edited_endings', None)
        
        # Check if edited_ending is properly formatted or missing
        if edited_endings is None:
            edited_endings = [["Missing edited_endings"]]
        elif not isinstance(edited_endings, list) or (isinstance(edited_endings, list) and not all(isinstance(e, list) for e in edited_endings)):
            edited_endings = [edited_endings] if isinstance(edited_endings, list) else [[edited_endings]]
        
        # Debugging: Print the keys and input components for inspection
        print(f"Keys in the row: {row.keys()}")
        print(f"Input components: premise='{premise}', initial='{initial}', original_ending='{original_ending}', counterfactual='{counterfactual}'")
        print(f"Output sequence(s): {edited_endings}")
        
        # Returning the individual components as separate items
        return pd.Series({
            'premise': premise,
            'initial': initial,
            'original_ending': original_ending,
            'counterfactual': counterfactual,
            'edited_endings': edited_endings # Keep edited_endings as a list
        })
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")
        logger.error(f"Problematic data row: {row}")
        # Return a series with error messages
        return pd.Series({
            'premise': "Error in premise",
            'initial': "Error in initial",
            'original_ending': "Error in original_ending",
            'counterfactual': "Error in counterfactual",
            'edited_endings': "Error in output"
        })

def collate_fn(batch, tokenizer):
    print("--- collate_fn: Starting ---")
    # Initialize containers for the batch's tokenized data
    tokenized_batch = {
        'input_ids': {'concatenated': []},
        'attention_mask': {'concatenated': []},
        'labels': []
    }

    for item in batch:
        # Concatenate text components for input_ids and attention_mask, except 'edited_endings'
        concatenated_text = f"{item['premise']} {item['initial']} {item['original_ending']} {item['counterfactual']}"
        tokenized = tokenizer(concatenated_text, padding='max_length', truncation=True, return_tensors="pt")
        tokenized_batch['input_ids']['concatenated'].append(tokenized['input_ids'].squeeze(0))
        tokenized_batch['attention_mask']['concatenated'].append(tokenized['attention_mask'].squeeze(0))

        # Handle 'edited_endings': Flatten, tokenize, and append
        if 'edited_endings' in item and item['edited_endings']:
            for group in item['edited_endings']:
                for sentence in group:
                    tokenized_endings = tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt")
                    tokenized_batch['labels'].append(tokenized_endings['input_ids'].squeeze(0))  # Assuming labels are individual sentences
        else:
            print("Missing edited_endings in item.")

    # Convert lists to tensors
    for key in ['input_ids', 'attention_mask']:
        tokenized_batch[key]['concatenated'] = torch.stack(tokenized_batch[key]['concatenated'])
    tokenized_batch['labels'] = torch.stack(tokenized_batch['labels']) if tokenized_batch['labels'] else None

    print("collate_fn: Batch prepared.")
    return tokenized_batch
      
