# src/models/model_T5.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
#from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_scorer
import logging
# evaluation
from sacrebleu import corpus_bleu
#from sacrerouge.metrics import Rouge
from bert_score import score as bert_score
#import spacy
#from wmd import WMD

import tempfile
import files2rouge

# Configurations and logger initialization
from src.utils.config import CONFIG 
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    Fine-tunes the FLAN-T5 model on a given dataset for narrative text generation tasks.
    """

    def __init__(self, model_name):
        """
        Initializes model, tokenizer, and evaluation metrics.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Initialize RougeScorer with all types of ROUGE metrics
        #rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        #self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        #self.rouge_types = rouge_types

        # Initialize a list to store outputs for each validation step
        self.current_val_step_outputs = []
    
    
    def forward(self, input_ids, attention_mask, labels):
        """
        Forward pass through the model. Calculates loss if labels are provided.
        """       
        print("--forward pass--")
        
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        
        # Pass the concatenated input_ids, attention_mask, and labels (if provided) to the model for processing.
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # If labels were provided, the model's output will include loss for training
        if labels is not None:
            print("Loss from model output:", output.loss.item())
        else:
            print("Model output generated without calculating loss (inference mode).")
        
        return output


    def training_step(self, batch, batch_idx):
        """
        Processes one batch of data during the training phase.
        """
        print("--training_step --") 
           
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
 
    
    def validation_step(self, batch, batch_idx):
        """
        Processes one batch of data during the validation phase. Calculates metrics.
        """
        
        print("-- validation_step --")   
        # Forward pass to compute loss and model outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Store necessary components for later metric calculation
        self.current_val_step_outputs.append({
            'generated_texts': generated_texts,
            'reference_texts': {
                'edited_endings': batch['edited_ending'],
                'premises': batch['premise'],
                'counterfactuals': batch['counterfactual'],
                'original_endings': batch['original_ending'],
            }
        })


    
    import subprocess
    import tempfile

    def calculate_metrics(self, generated_texts, reference_texts):
        # Convert lists of generated and reference texts into strings
        generated_text_str = "\n".join(generated_texts)
        reference_text_str = "\n".join(["\n".join(refs) for refs in reference_texts])

        # Write the strings to temporary files
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as gen_file, \
            tempfile.NamedTemporaryFile(delete=False, mode='w') as ref_file:
            gen_file.write(generated_text_str)
            ref_file.write(reference_text_str)
            gen_file_path = gen_file.name
            ref_file_path = ref_file.name

        # Use subprocess to call files2rouge
        cmd = f'files2rouge {ref_file_path} {gen_file_path}'
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            output = e.output  # If there's an error, capture the output

        # Here you would parse the output from files2rouge to extract the ROUGE scores
        # The parsing will depend on the exact format of the files2rouge output
        # Example: rouge_scores = parse_files2rouge_output(output)

        # Don't forget to calculate your BLEU and BERT scores as before
        # Example:
        # bleu_score = calculate_bleu(generated_texts, reference_texts)
        # bert_scores = calculate_bert(generated_texts, reference_texts)

        # Return a dictionary of your metrics
        return {
            'rouge': rouge_scores,  # Adjust this based on your actual parsing results
            'bleu': bleu_score,
            'bert_score': bert_scores
        }

    # Ensure to define or adjust functions like `parse_files2rouge_output`, `calculate_bleu`, and `calculate_bert`
    # according to your needs and the actual output format of files2rouge, BLEU, and BERT scoring functions.

    def parse_rouge_output(output):
        # Placeholder function for parsing the ROUGE output from files2rouge
        # You'll need to implement this based on the actual output format
        rouge_scores = {}
        # Parsing logic here
        return rouge_scores



    def on_validation_epoch_end(self):
        """
        Organizes and aggregates the metrics for different types of reference texts, 
        then logs them using PyTorch Lightning's logging mechanism.
        """
        # Initialize a dictionary to aggregate metrics for different types of reference texts.
        aggregated_metrics = {}

        # Loop over each output collected during the validation steps.
        for output in self.current_val_step_outputs:
            # Each output contains generated texts and their corresponding reference texts for different types.
            for ref_type, refs in output['reference_texts'].items():
                # Check if this type of reference text has already been encountered and initialized in the aggregator.
                if ref_type not in aggregated_metrics:
                    # If not, initialize an entry for this type with empty lists for generated and reference texts.
                    aggregated_metrics[ref_type] = {'generated_texts': [], 'reference_texts': []}
                
                # Extend the lists of generated and reference texts for this type with the current output's data.
                aggregated_metrics[ref_type]['generated_texts'].extend(output['generated_texts'])
                aggregated_metrics[ref_type]['reference_texts'].extend(refs)

        # Now, for each type of reference text, calculate and log the metrics.
        for ref_type, data in aggregated_metrics.items():
            # Calculate metrics using the aggregated generated and reference texts for this type.
            metrics = self.calculate_metrics(data['generated_texts'], data['reference_texts'])
            
            # Iterate over each metric calculated for this type of reference text.
            for metric_name, metric_value in metrics.items():
                # Log the metric value using PyTorch Lightning's logging mechanism.
                # This includes the type of reference text and the metric name for clarity.
                # For example, if ref_type is 'edited_endings', and the metric is BLEU, it logs as 'edited_endings_bleu'.
                self.log(f'{ref_type}_{metric_name}', metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear the list of outputs to prepare for the next validation epoch.
        # This is necessary to prevent carrying over data from the previous epochs.
        self.current_val_step_outputs = []


    def test_step(self, batch, batch_idx):
        """
        Processes one batch of data during the testing phase. Reuses validation logic.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


    def generate_text(self, input_ids, attention_mask):
        """
        Generates text based on input_ids and attention_mask.
        """
        print("-- generate_text --") 

        # Generate a tensor of token IDs based on the input_ids and attention_mask
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Decode the generated token IDs back into human-readable text
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts
    
    def configure_optimizers(self):
        """
        Configures the model's optimizer.
        """
        print("-- configure_optimizers --") 
        
        #lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        #return torch.optim.AdamW(self.model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer