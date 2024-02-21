#/src/data_loader.py 

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from src.utils.utils import preprocess_data, collate_fn
from functools import partial

class CustomJSONDataset(Dataset):
    """
    A PyTorch Dataset class that handles loading and preprocessing data from JSON files.
    """

    def __init__(self, file_path, tokenizer):
        print(f"Initializing dataset with file: {file_path}")
        try:
            data = pd.read_json(file_path, lines=True)
            print(f"Successfully read {len(data)} rows from {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.tokenizer = tokenizer

        # Pass tokenizer to preprocess_data via lambda function
        self.processed_data = data.apply(lambda row: preprocess_data(row, tokenizer), axis=1, result_type='expand') # Use lambda to pass tokenizer
        print(f"Data preprocessing completed. Processed data size: {len(self.processed_data)}")


    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index from the dataset.
        """
        print(f"---Dataloader getitem---")
        
        item = self.processed_data.iloc[idx]
        # Debugging: Only print for the first few indices
        if idx < 3:  #
            print(f"Item at index {idx}: {item.to_dict()}")
            print(f"Keys at index {idx}: {item.keys().tolist()}")
        return item

def create_dataloaders(data_path, file_names, batch_size, tokenizer, num_workers=0):
    """
    Creates a dictionary of DataLoader objects for each specified JSON file.
    """
    dataloaders = {}
    for file_name in file_names:
        # Construct the full path to the JSON file
        file_path = Path(data_path) / file_name
        # Raise an error if the file doesn't exist
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        # Create a custom dataset using the JSON file and the preprocess_data function directly
        dataset = CustomJSONDataset(file_path, tokenizer)  # Pass tokenizer here

        # Create a DataLoader for batching and loading the dataset
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,  
            num_workers=num_workers
        )

        # Store the DataLoader in the dictionary using the file name as the key
        dataloaders[file_name] = dataloader
        print(f"Dataloader created for {file_name}")
    
    return dataloaders