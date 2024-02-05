# src/utils/utils.py

import json
import pandas as pd
import logging
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
        dict: A dictionary with tokenized inputs and outputs, ready for model training or inference.
    """
    # Fetch the separator token from the CONFIG or use a default
    separator_token = CONFIG.get("separator_token", "<s>")
    
    # Constructing the input and output sequences for each item in the batch
    input_sequences = []
    output_sequences = []
    for item in batch:
        input_sequence = f"{item['premise']} {item['initial']} {item['original_ending']} {separator_token} {item['initial']} {item['counterfactual']}"
        input_sequences.append(input_sequence)
        output_sequences.append(item['edited_ending'])
    
    # Print keys and the first input sequence for inspection
    print("Keys in the batch during collate_fn:", batch[0].keys())
    if input_sequences:  # Checking if input_sequences is not empty
        print("First input sequence after concatenation:", input_sequences[0])
    
    # Tokenizing input sequences
    encoding = tokenizer(input_sequences, padding=True, truncation=True, return_tensors="pt")
    
    # Tokenizing output sequences (edited_endings)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(output_sequences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    
    # Adding labels to the encoding
    encoding["labels"] = labels
    
    # Print keys in the final batch after tokenization
    print("Keys in the batch after tokenization:", encoding.keys())
    return encoding
