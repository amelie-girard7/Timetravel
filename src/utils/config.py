#src/utils/config.py

import os
from pathlib import Path

# Allow the root directory to be set via an environment variable for flexibility
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))
BARTSCORE_DIR = ROOT_DIR / "src" / "BARTScore_metric"

# Configuration parameters
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,  # Adding root_dir here
    "data_dir": ROOT_DIR / "data",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "bart_score_dir": BARTSCORE_DIR,  # Add the BARTScore directory
    
    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    #"model_name": os.getenv('MODEL_NAME', "google/flan-t5-large"),
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Convert to int as environment variables are strings
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 3)),
    "learning_rate": 2e-5,
}

# Optionally, validate or create the directories
for path_key in ['data_dir', 'models_dir', 'logs_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
