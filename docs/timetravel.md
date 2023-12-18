# Summary of the Paper "Counterfactual Story Reasoning and Generation"

[Counterfactual Story Reasoning and Generation](https://arxiv.org/abs/1909.04076)

**Objective**: 
- The paper introduces the task of Counterfactual Story Rewriting. It's about revising an original story to align with a counterfactual event, requiring an understanding of causal narrative chains and counterfactual invariance.

**Challenges**:
- Difficulty in discriminating between reasonable and unreasonable counterfactuals.
- Risk of models exploiting dataset artifacts rather than robustly reasoning about counterfactuals.
- Need for models to understand underlying dynamics and reason about causal invariance.
- Struggle of neural language models to rewrite endings consistently.

**Dataset**: 
- TIMETRAVEL dataset with 29,849 counterfactual rewritings.
- Collected through a two-step task with Amazon Mechanical Turk crowdworkers.
- Aimed to encourage models to understand situation dynamics and generate consequences of counterfactual reasoning.

**Models Evaluated**: 
- Performance of pretrained language models (GPT, GPT-2 variants) evaluated.
- Settings: zero-shot, unsupervised, and supervised.
- Effectiveness of metrics like BLEU, ROUGE, BERTScore analyzed.

**Methodology**: 
- Training settings included unsupervised training, fine-tuning on ROCStories corpus, and supervised training with human-annotated endings.
- Two groups of crowdworkers involved in annotation: creating counterfactuals and coherent story endings.

### Step 2: Preparing to Replicate Using LAMA2

#### 2.1. Select an Open-Source LLM
- **Action**: Choose - LAMA2 from Hugging Face. (Too big for IHCP: Try NECTAR)
                     -  flan-T5

#### 2.2. Obtain the TIMETRAVEL Dataset
- **Action**: Download the dataset from the provided source.(src\timetravel\data)

#### 2.3. Set Up the Environment
- **Tools**: Pytorch Lightning, IHPC (heaphesots3),

#### 2.4. Preprocess the Data
- **Data Files**: Include supervised (small and large sets), unsupervised, dev, and test sets.
- **Format**: Adapt the data format to LAMA2's requirements.

- **Splitting**: Ensure data is split as per the paper (training, validation, test sets).

#### 2.5. Model Training
- **Zero-Shot Setting**: Test LAMA2's base performance without specific training.
- **Unsupervised Training**: Train on ROCStories corpus without explicit counterfactual annotations.
- **Supervised Training**: Fine-tune with human-annotated counterfactual rewritings.

#### 2.6. Evaluation
- **Metrics**: Implement BLEU, ROUGE, BERTScore for evaluation.
- **Comparison**: Compare model performance across different settings with the paper's results.

#### 2.7. Analysis and Reporting
- **Analysis**: Assess model's counterfactual reasoning and narrative consistency.
- **Documentation**: Note deviations from the paper's results and hypothesize reasons.

### Step 3: Implementation in Python

Now, we'll move to the practical part, which involves writing and running Python code to achieve the steps outlined above. This includes setting up the environment, preprocessing the data, training the model, and evaluating its performance.

The actual coding and running of scripts can be done in an IDE like VS Code or directly in a Google Colab notebook. Remember, the success of replication depends on accurately following the methodology of the paper and adapting it to work with LAMA2.

---


Let's review the content of the three files(`main.py`, `model.py`, `utils.py`) to understand the structure of your program and identify the cause of the error when executing `main.py`.

### 1. `main.py`

This file is the entry point of your program. It likely contains the script to process the data, create the model, train it, and save the trained model. It probably also sets up the necessary configurations, loads the data, and initiates the training process.

### 2. `model.py`

This file should contain the definition of your PyTorch Lightning model class, `FlanT5FineTuner`. This class would typically include methods for the forward pass, training, validation, test steps, and optimizer configuration. It's crucial for defining how your model will operate during training and evaluation.

### 3. `utils.py`

This file is expected to contain utility functions like `count_json_lines`, `load_first_line_from_json`, and `preprocess_data`. These functions assist in handling and processing the JSON data files. Additionally, `collate_fn` should be defined here, which is responsible for preparing batches of data during training.

To understand the error you're encountering, I will review the contents of these files and provide insights or suggestions for fixes. Let me start by examining `main.py`.

The `main.py` file contains the main script for your program. Here's a summary of its structure:

1. **Imports**: The script imports necessary modules and functions from `utils.py` and `model.py`.

2. **Configuration**: A `CONFIG` dictionary is defined, containing key parameters like the root directory, model name, batch size, number of workers, and maximum epochs.

3. **Main Function (`main()`)**:
   - Sets up paths for data and model saving.
   - Iterates over the dataset files (`train_supervised_small1.json`, `dev_data1.json`, `test_data1.json`), reads them, and processes the first line for display.
   - Reads and preprocesses the data using the `preprocess_data` function.
   - (The rest of the script is not visible in the provided snippet, but likely includes model initialization, data loading, training, and model saving.)

Next, let's review the contents of `model.py` to understand the `FlanT5FineTuner` class implementation.

The `model.py` file defines the `FlanT5FineTuner` class, a PyTorch Lightning model for fine-tuning the Flan-T5 model. Here's an overview:

- **Imports**: The script imports PyTorch, the Transformers library, and PyTorch Lightning.

- **Class Definition (`FlanT5FineTuner`)**:
  - **Initialization (`__init__`)**: Initializes the T5 model and tokenizer using the provided `model_path`.
  - **Forward Method**: Defines the forward pass of the model.
  - **Training Step**: Implements the training step, computing the loss.
  - **Validation Step**: Implements the validation step, computing the validation loss.
  - **Test Step**: Implements the test step, computing the test loss.
  - **Optimizer Configuration**: Configures the optimizer for the model.

Finally, let's review the content of `utils.py` to see the utility functions and the implementation of `collate_fn`.

The `utils.py` file contains utility functions and the `collate_fn` function. Here's a summary:

- **Utility Functions**:
  - `count_json_lines(file_path)`: Counts the number of lines in a JSON file.
  - `load_first_line_from_json(file_path)`: Loads the first line from a JSON file.
  - `preprocess_data(row)`: Preprocesses a row of the dataset for the T5 model. It combines 'premise', 'initial', and 'counterfactual' into a single string and uses the first 'edited_ending' as the target output.

- **Collate Function (`collate_fn`)**:
  - Accepts a batch of data and a tokenizer.
  - Tokenizes input and output texts for the T5 model.
  - Prepares the tokenized data for training by setting up the encoding and labels.
  - Returns the prepared batch for training.
