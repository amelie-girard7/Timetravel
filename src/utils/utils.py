import json
import pandas as pd

def count_json_lines(file_path):
    """
    Count the number of lines in a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_first_line_from_json(file_path):
    """
    Load the first line from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        raise IOError(f"Error reading from {file_path}: {e}")

def preprocess_data(row):
    try:
        # Extracting fields from the row
        premise = row.get('premise', "Missing premise")
        initial = row.get('initial', "Missing initial")
        original_ending = row.get('original_ending', "Missing original_ending")
        counterfactual = row.get('counterfactual', "Missing counterfactual")
        edited_ending = row.get('edited_ending', ["Missing edited_ending"])
        
        # Ensure edited_ending is a list
        if not isinstance(edited_ending, list):
            edited_ending = ["Invalid format for edited_ending"]
        
        # Constructing the input sequence (Premise, Initial, Original Ending, Counterfactual)
        separator_token = "[s]"
        input_sequence = f"{premise} {separator_token} {initial} {separator_token} {original_ending} {separator_token} {counterfactual}"
        
        # Constructing the output sequence (Edited Ending)
        output_sequence = ' '.join(edited_ending)
        
        return pd.Series({
            'input_ids': input_sequence,
            'output_ids': output_sequence
        })
    except Exception as e:
        print(f"An error occurred while processing the data: {e}")
        print(f"Problematic data row: {row}")
        return pd.Series({
            'input_ids': "Error in input",
            'output_ids': "Error in output"
        })

def collate_fn(batch, tokenizer):
    """
    Tokenize and collate a batch of data for the T5 model.
    """
    # Extracting input and output sequences from the batch
    input_sequences = [item['input_ids'] for item in batch]
    output_sequences = [item['output_ids'] for item in batch]
    
    # Tokenizing input sequences
    encoding = tokenizer(input_sequences, padding=True, truncation=True, return_tensors="pt")
    
    # Tokenizing output sequences
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(output_sequences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    
    # Adding labels to the encoding
    encoding["labels"] = labels
    
    return encoding