# src/utils/utils.py

import json
import logging
import nltk
from nltk.corpus import wordnet as wn
import torch
import torch.nn.utils.rnn
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

def count_json_lines(file_path):
    """
    Counts the number of lines in a JSON file, which is useful for estimating
    the dataset size or for iterative processing without loading the entire file.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        int: The number of lines in the file.
    """
    logger.info(f"Counting lines in file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
def load_first_line_from_json(file_path):
    """
    Loads and parses the first line from a JSON file. This is useful for inspecting
    the data structure without loading the entire file.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The first JSON object in the file.
    """
    logger.info(f"Loading first line from JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {e}")
        raise IOError(f"Error reading from {file_path}: {e}")


def preprocess_data(row, tokenizer):
    """
    Prepares a single row of data for model input by tokenizing the text fields.
    It constructs the input sequence by combining story parts and tokenizes them.
    
    Parameters:
        row (pd.Series): A row from the dataset.
        tokenizer: The tokenizer used for text processing.
        max_length (int): The maximum token length for model inputs.
        
    Returns:
        dict: A dictionary containing tokenized inputs, attention masks, labels, and original text for evaluation.
    """
    logger.debug("Preprocessing data row...")
    
    try:
        # Define the separator token specific to the T5 model.
        separator_token = "</s>"
        
        # Construct the input sequence with all components separated by the model-specific separator token.
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']}"
            f"{row['original_ending']} {separator_token} "
            f"{row['premise']} {row['counterfactual']}"
        )
        
        # Tokenize the input sequence with truncation to max_length and no padding here.
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        
        # Join the list of edited endings into a single string
        edited_ending_joined = ' '.join(row['edited_ending'])
        
        # Tokenize the output sequence (edited ending) with truncation to max_length.
        tokenized_ending = tokenizer.encode_plus(
            edited_ending_joined, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        
        # Return the tokenized inputs, labels, and original data fields for evaluation.
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            # Include non-tokenized data for metric calculations.
            'premise': row['premise'],
            'initial': row['initial'],
            'original_ending': row['original_ending'],
            'counterfactual': row['counterfactual'],
            'edited_ending': edited_ending_joined
        }
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None
    
def collate_fn(batch, pad_token_id=0,attention_pad_value=0):
    """
    Collates a batch of preprocessed data into a format suitable for model input,
    including padding to equalize the lengths of sequences within the batch.
    
    Parameters:
        batch (list of dicts): A batch of data points.
        pad_token_id (int, optional): Token ID used for padding. Default is 0.
        
    Returns:
        dict: A dictionary containing batched and padded input_ids, attention_mask,
        labels, and other fields for evaluation.
    """
    
    # Unpack the batch into separate lists for each field.
    input_ids, attention_mask, labels, premise, initial, original_ending, counterfactual, edited_ending = list(zip(*batch))

    # Padding sequences for 'input_ids', 'attention_masks', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=attention_pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    
    # Return the padded tensors along with the additional fields for evaluation.
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }