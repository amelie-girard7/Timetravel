#/src/data_loader.py 

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from src.utils.utils import preprocess_data, collate_fn

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
    This function is responsible for creating DataLoader instances for datasets.
    """
    dataloaders = {}
    for file_name in file_names:
        # Construct the full path to the dataset file
        file_path = Path(data_path) / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        # Initialize a custom dataset. This step involves reading the data file and preprocessing it
        # using the tokenizer to make it suitable for model input.
        dataset = CustomJSONDataset(file_path, tokenizer)

        # Create a DataLoader for the dataset. DataLoader abstracts the complexity of fetching,
        # transforming, and batching the data, making it ready for training or validation.
        # `collate_fn` is used to specify how a list of samples is combined into a batch.
        # This is especially important because we are dealing with variable-length inputs.
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            collate_fn=collate_fn,  # Custom function to combine data samples into a batch
            num_workers=num_workers,  # Number of subprocesses for data loading. 0 means data will be loaded in the main process.
            
            # `persistent_workers=True` is recommended when using multiple workers (num_workers > 0).
            # It keeps the worker processes alive across data fetches rather than restarting them for each fetch.
            # This can lead to significant performance improvements, especially for large datasets or complex
            # preprocessing pipelines, as it reduces the overhead from constantly creating and destroying worker processes.
            # However, it's only effective (and only makes sense to enable) when `num_workers` is greater than 0.
            # When there are no worker processes (num_workers=0), this setting has no effect.
            persistent_workers=True if num_workers > 0 else False,
        )

        # Store the DataLoader in a dictionary using the file name as the key.
        # This allows for easy access to different dataloaders for training, validation, and testing.
        dataloaders[file_name] = dataloader
        print(f"Dataloader created for {file_name}")

    return dataloaders

