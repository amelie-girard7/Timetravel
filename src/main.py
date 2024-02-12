# src/main.py
import os
import sys
import torch
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# Append src to the system path for imports.
# This allows the script to access the 'src' directory as if it were a package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importing custom modules.
from src.models.model_T5 import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.utils import preprocess_data, collate_fn
from src.utils.config import CONFIG

# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model():
    """
    Initializes and returns the Flan T5 model configured with the specified model name.
    
    Returns:
        FlanT5FineTuner: The initialized Flan T5 model.
    """
    model = FlanT5FineTuner(CONFIG["model_name"])
    return model

def setup_dataloaders(model):
    """
    Sets up PyTorch dataloaders for the training, validation, and test datasets.
    
    Parameters:
        model (FlanT5FineTuner): The model for which the dataloaders are being set up.
    
    Returns:
        dict: A dictionary containing the training, validation, and test dataloaders.
    """
    data_path = CONFIG["data_dir"] / 'transformed'
    file_names = ['train_supervised_small_sample.json', 'dev_data_sample.json', 'test_data_sample.json']

    dataloaders = create_dataloaders(
        data_path,
        file_names,
        CONFIG["batch_size"],
        model.tokenizer,  # Pass tokenizer directly to create_dataloaders
        CONFIG["num_workers"]
    )
    # Code to check the first batch from each dataloader
    for file_name, dataloader in dataloaders.items():
        for batch in dataloader:
            print(f"First batch from {file_name}:")
            print(batch)
            break  # Break after printing the first batch
    return dataloaders

def setup_trainer(model_save_path):
    """
    Configures the PyTorch Lightning trainer with checkpointing and TensorBoard logging.
    
    Parameters:
        model_save_path (Path): Path where the model checkpoints will be saved.
    
    Returns:
        Trainer: The configured PyTorch Lightning trainer.
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    logger = TensorBoardLogger(save_dir=os.path.join(model_save_path, 'lightning_logs'), name='flan-t5')

    # Define a PyTorch Lightning trainer with the desired settings.
    if torch.cuda.is_available():
        trainer = Trainer(
            max_epochs=CONFIG["max_epochs"],
            accelerator='gpu',
            devices=1,  # Specify the number of GPUs you want to use.
            logger=logger,
            callbacks=[checkpoint_callback],
            # Additional parameters can be passed according to requirements.
        )
    else:
        trainer = Trainer(
            max_epochs=CONFIG["max_epochs"],
            logger=logger,
            callbacks=[checkpoint_callback],
            # Additional parameters can be passed according to requirements.
        )
    return trainer

def main():
    """
    The main execution function for setting up and running the training process.
    Handles the entire workflow from model setup, data loading, training, and testing.
    """
    try:
        model = setup_model()
        dataloaders = setup_dataloaders(model)
        model_save_path = CONFIG["models_dir"]
        model_save_path.mkdir(exist_ok=True)

        trainer = setup_trainer(model_save_path)

        train_dataloader = dataloaders['train_supervised_small_sample.json']
        valid_dataloader = dataloaders['dev_data_sample.json']
        test_dataloader = dataloaders['test_data_sample.json']

        # Fetch the first batch from the training DataLoader
        batch = next(iter(train_dataloader))

        # Print out each key with the corresponding data shape, type, and a preview of the first few items for inspection
        for key in batch.keys():
            print(f"{key}:")
            print(f"  - Type: {type(batch[key])}")
            print(f"  - Dtype: {batch[key].dtype}")
            print(f"  - Shape: {batch[key].shape}")
            # Only print a preview if the key refers to tensor data
            if torch.is_tensor(batch[key]):
                print(f"  - Data (preview): {batch[key][:1]}")
            else:
                print(f"  - Data (preview): {batch[key][:1].tolist()}")

        # Start the training process.
        trainer.fit(model, train_dataloader, valid_dataloader)
        
        # Evaluate the model on the test data after training.
        trainer.test(dataloaders=test_dataloader)
    except Exception as e:
        logger.exception(f"An error occurred during training or testing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

