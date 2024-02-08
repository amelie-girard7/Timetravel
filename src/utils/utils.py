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
    try:
        # Extract fields from the row
        premise = row.get('premise', "Missing premise")
        initial = row.get('initial', "Missing initial")
        original_ending = row.get('original_ending', "Missing original_ending")
        counterfactual = row.get('counterfactual', "Missing counterfactual")
        edited_ending = row.get('edited_ending', ["Missing edited_ending"])
        
        # Ensure edited_ending is a list
        if not isinstance(edited_ending, list):
            edited_ending = ["Invalid format for edited_ending"]
        
        # Constructing the output sequence (Edited Ending)
        output_sequence = ' '.join(edited_ending)
        
        # Print to inspect if all keys are present
        print("Keys in the row during preprocess_data:", row.keys())
        
        print(f"Input components: premise={premise}, initial={initial}, original_ending={original_ending}, counterfactual={counterfactual}")
        print(f"Output sequence: {output_sequence}")
        
        # Returning the individual components as separate items
        return pd.Series({
            'premise': premise,
            'initial': initial,
            'original_ending': original_ending,
            'counterfactual': counterfactual,
            'edited_ending': output_sequence  # Output sequence is a single string
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
            'edited_ending': "Error in output"
        })

def collate_fn(batch, tokenizer):
    """
    Tokenize and collate a batch of data for the T5 model. 
    
    Args:
        batch (list): A list of samples to be collated.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the text data.
    
    Returns:
        dict: A dictionary with tokenized inputs for each component and outputs, ready for model training or inference.
    """
    tokenized_batch = {}
    
    # Tokenize each component individually and store in the tokenized_batch dictionary
    for component in ['premise', 'initial', 'original_ending', 'counterfactual', 'edited_ending']:
        component_sequences = [item[component] for item in batch]
        tokenized_batch[component] = tokenizer(component_sequences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    
    # Construct attention_mask for input_ids (optional, based on your model's requirements)
    attention_masks = [torch.ones_like(tokenized_batch[component], dtype=torch.long) for component in tokenized_batch if component != 'edited_ending']
    tokenized_batch['attention_mask'] = torch.cat(attention_masks, dim=1)
    
    return tokenized_batch
