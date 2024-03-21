#src/utils/config.py

import os
from pathlib import Path

# Allow the root directory to be set via an environment variable for flexibility
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))
BARTSCORE_DIR = ROOT_DIR / "src" / "BARTScore_metric"

# Configuration parameters
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR, 
    "data_dir": ROOT_DIR / "data",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "bart_score_dir": BARTSCORE_DIR,
    
    # File names
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",
    
    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),
    
    # Text generation parameters
    "max_gen_length": 250,

    # Evaluation metrics settings
    "eval_batch_size": 4,
    
    # BERTScorer settings
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 4,

    # BARTScorer settings
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",
}

# Optionally, validate or create the directories
for path_key in ['data_dir', 'models_dir', 'logs_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True) 
