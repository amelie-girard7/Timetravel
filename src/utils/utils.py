# src/utils/utils.py

from fileinput import filename
import json
import pandas as pd
import logging

import torch
from src.utils.config import CONFIG
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer
PAD_TOKEN_ID = 0
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
"""
def preprocess_data(row):
    
    Preprocess a single row of data to construct the input and output sequences for the model.
    
    Args:
        row (pd.Series): A pandas Series representing a single row of data.
    
    Returns:
        pd.Series: A pandas Series containing the processed input and output sequences.
    
    print(f"--- Preprocessing Dataset ---")
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
"""
     

"""
        
def collate_fn(batch, tokenizer):
    
    Tokenize and collate a batch of data for the T5 model.
    
    Args:
        batch (list): A list of samples to be collated.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the text data.
    
    Returns:
        dict: A dictionary with tokenized inputs for each component and outputs, ready for model training or inference.
    
    print("Starting collation of batch data")
    tokenized_batch = {}
    
    # Tokenize each component individually and store in the tokenized_batch dictionary
    for component in ['premise', 'initial', 'original_ending', 'counterfactual', 'edited_ending']:
        component_sequences = [item[component] for item in batch]  # Extract sequences for each component
        tokenized_batch[component] = tokenizer(component_sequences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        # tokenized_batch[component] = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt").input_ids
    
    # Excludes edited_ending from the attention mask calculation. It generates a mask of ones for all components except edited_ending and concatenates these masks.
    attention_masks = [torch.ones_like(tokenized_batch[component], dtype=torch.long) for component in tokenized_batch if component != 'edited_ending']
    tokenized_batch['attention_mask'] = torch.cat(attention_masks, dim=1)
    
    print("Completed collation for current batch")
    return tokenized_batch
"""

def preprocess_data(row, tokenizer, max_length=512):
    """
    Preprocesses a single data row for T5 model input.
    """
    try:
        # Define separator tokens and model-specific formatting.
        separator_token = "</s>"
        
        # Construct the input sequence with proper formatting.
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']}"
            f"{row['original_ending']} {separator_token} "
            f"{row['premise']} {row['counterfactual']}"
        )
        # TODO remove the padding : Why does it throw and error?
        # Tokenize input_sequence and the edited_ending.
        # TODO: Understand the max_seq_len to apply trunctation
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=max_length
        )
        
        # Constructing the output sequence (Edited Ending)
        edited_ending_joined = ' '.join(row['edited_ending'])
        
        tokenized_ending = tokenizer.encode_plus(
            edited_ending_joined, truncation=True, return_tensors="pt", max_length=max_length
        )
        
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            # Include non-tokenized data for metric calculations if necessary.
            'premise': row['premise'],
            'initial': row['initial'],
            'original_ending': row['original_ending'],
            'counterfactual': row['counterfactual'],
            'edited_ending': edited_ending_joined
        }
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None


def collate_fn(batch):
    """
    Collate function to process a batch of data and return the required
    tensor formats for the model.
    
    Args:
        batch (list): A list of dictionaries with keys 'input_ids', 'attention_mask', 'labels', etc.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding the texts.
    
    Returns:
        dict: A dictionary with keys 'input_ids', 'attention_mask', 'labels', etc.
    """
    print("Running custom collate function on batch of size {}".format(len(batch)))
    
    input_ids, attention_mask, labels, premise, initial, original_ending, counterfactual, edited_ending = list(zip(*batch))
    # TODO: Try to pass the PAD_TOKEN_ID as a parameter to the collate_fn
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_TOKEN_ID)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        # Additional fields for evaluation
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }
