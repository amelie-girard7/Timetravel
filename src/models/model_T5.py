# src/models/model_T5.py
import sys
import torch
from src.utils.config import CONFIG
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer

# Add BARTScore directory to Python path
bart_score_path = str(CONFIG["bart_score_dir"])
if bart_score_path not in sys.path:
    sys.path.append(bart_score_path)
    
from src.BARTScore_metric.bart_score import BARTScorer

import logging
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    This class defines a PyTorch Lightning model for fine-tuning a Flan-T5 model.
    It handles the forward pass, training, validation, and testing steps.
    """

    def __init__(self, model_name):
        """
        Initializes the model components, tokenizer, and metrics scorer.
        
        Args:
            model_name (str): The name of the T5 model to be used.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Initialise sacre bleu and Rouge Bert
        self.sacre_bleu = BLEU()      
        self.rouge = Rouge()
        self.bert_scorer = BERTScorer(model_type='roberta-base-mnli', device='cuda:0', num_layers=None, batch_size=4)
    
        
        # Initialize BARTScorer : TODO (Replace with the actual path)
        self.bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        self.bart_scorer.load(path='path_to_trained_bartscore_model')

    
    def forward(self, input_ids, attention_mask, labels):
        """
        Performs the forward pass of the model              
        Returns:
            The output from the T5 model, which includes loss when labels are provided, and logits otherwise.
        """
        print("--forward pass--")
        
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        
        # Pass the concatenated input_ids, attention_mask, and labels  to the model.
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # If labels were provided, the model's output will include loss for training.
        if labels is not None:
            print("Loss from model output:", output.loss.item())
        else:
            print("Model output generated without calculating loss (inference mode).")
        
        return output


    def training_step(self, batch, batch_idx):
        """  
        Defines the training logic for a single batch, where a forward pass is performed and the loss is calculated.
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
        Performs a validation step for a single batch.
        It calculates the loss, generates predictions, and prepares data for metric calculation.
        """
        print("-- validation_step --")
        
        # Ensure model is in evaluation mode
        # This disables dropout or batchnorm layers and is important for
        # model evaluation to ensure consistent results
        self.model.eval()
           
        # Forward pass to compute loss and model outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print(f"Validation loss: {val_loss.item()}")

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=250 # Specify the desired maximum length of generated text
        )
        
        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text.
        edited_endings = batch['edited_ending']

        #printing details for the first story in the batch
        print(f"--Premise: {batch['premise'][0]}")
        print(f"--Initial: {batch['initial'][0]}")
        print(f"--Counterfactual: {batch['counterfactual'][0]}")
        print(f"--Original Ending: {batch['original_ending'][0]}")
        print(f"--Edited Ending: {edited_endings[0]}")
        print(f"--Generated Text: {generated_texts[0]}")  


        # Store output information for metric calculation at the end of the epoch
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            # Add other story components for later use in metric calculations
            'premises': batch['premise'],
            'counterfactuals': batch['counterfactual'],
            'original_endings': batch['original_ending'],
            'initials': batch['initial'],
        }
        self.current_val_step_outputs.append(output)
        print(f"Validation step {batch_idx} completed.")

        
    def on_validation_epoch_end(self):
        """
        Handles operations to perform at the end of each validation epoch.
        """
        print("-- on_validation_epoch_end --")
        self.model.train()
        # Prepare lists to store generated texts and their corresponding references
        all_generated_texts = []
        all_edited_endings = []
        all_counterfactuals = []
        all_initials = []
        all_original_endings = []

        # Aggregate texts from the outputs
        for output in self.current_val_step_outputs:
            all_generated_texts.extend(output['generated'])
            all_edited_endings.extend(output['edited_endings'])
            all_counterfactuals.extend(output['counterfactuals'])
            all_initials.extend(output['initials'])
            all_original_endings.extend(output['original_endings'])
            
       
        print("Aggregated texts for BLEU and ROUGE  and BERT similarity score calculation.")

        # Calculate and log BLEU similarity scores for various comparisons
        self.calculate_and_log_bleu_scores(all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings)

        # Calculate and log ROUGE similarity scores for various comparisons
        self.calculate_and_log_rouge_scores(all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings)
        
        # Calculate and log BERT similarity scores for various comparisons 
        self.calculate_and_log_bert_similarity(all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings)
        
        # Calculate and log BART similarity scores
        self.calculate_and_log_bart_similarity(all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings)

        # Clear the list of outputs for the next epoch
        self.current_val_step_outputs = []
        print("Validation epoch ended. Metrics logged.")
        

    def calculate_and_log_bleu_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BLEU scores for generated texts against various reference components.
        """
        # Prepare reference lists for BLEU calculations
        edited_endings_refs = [[ending] for ending in all_edited_endings]
        counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        initials_refs = [[init] for init in all_initials]
        original_endings_refs = [[orig] for orig in all_original_endings]

        # Calculate and log BLEU scores for generated_text vs. story components (edited_ending, cunterfactual,initial and original_endings )
        all_comparisons = [
            ('bleu_prediction_edited', all_generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, counterfactuals_refs),
            ('bleu_prediction_initial', all_generated_texts, initials_refs),
            ('bleu_prediction_original', all_generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bleu_edited_ending_initial', all_edited_endings, all_initials),
            ('bleu_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Calculate and log BLEU scores for each comparison
        for label, texts, references in all_comparisons:
            try:
                # Directly calculate the BLEU score and assume it to be a float
                bleu_result = self.sacre_bleu.corpus_score(texts, references)
                # The bleu_result.score is already a float representing the BLEU score
                bleu_score = bleu_result.score  # This is correct and should not cause an issue

                # Log the BLEU score
                self.log(label, bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                print(f"{label}: {bleu_score}")
            except Exception as e:
                print(f"Error calculating {label}: {e}")


    def calculate_and_log_rouge_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs ROUGE scores for the generated texts against various components,
        and also for edited endings against other story components.
        """
        print("Calculating ROUGE scores...")
        # Original comparisons for generated texts
        all_comparisons = [
            ('rouge_prediction_edited', all_generated_texts, all_edited_endings),
            ('rouge_prediction_cf', all_generated_texts, all_counterfactuals),
            ('rouge_prediction_initial', all_generated_texts, all_initials),
            ('rouge_prediction_original', all_generated_texts, all_original_endings),
            ('rouge_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('rouge_edited_ending_initial', all_edited_endings, all_initials),
            ('rouge_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        for label, hypotheses, references in all_comparisons:
            rouge_scores = self.rouge.get_scores(hypotheses, references, avg=True)
            print(f"{label}: {rouge_scores}")

            # Log ROUGE scores
            for score_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                if score_type in rouge_scores:
                    self.log(f"{label}_{score_type}_f", rouge_scores[score_type]['f'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{label}_{score_type}_p", rouge_scores[score_type]['p'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{label}_{score_type}_r", rouge_scores[score_type]['r'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    print(f"{label}_{score_type} F1: {rouge_scores[score_type]['f']} Precision: {rouge_scores[score_type]['p']} Recall: {rouge_scores[score_type]['r']}")

    
    
    def calculate_and_log_bert_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BERT similarity scores for generated texts against various components,
        and also for edited endings against other story components.
        """
        print("Calculating BERT similarity scores...")
        
        # Define the original and edited comparisons
        all_comparisons = [
            ('bert_prediction_edited', all_generated_texts, all_edited_endings),
            ('bert_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bert_prediction_initial', all_generated_texts, all_initials),
            ('bert_prediction_original', all_generated_texts, all_original_endings),
            ('bert_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bert_edited_ending_initial', all_edited_endings, all_initials),
            ('bert_edited_ending_original', all_edited_endings, all_original_endings),
        ]
        # Calculate and log BERT similarity scores for each comparison
        for label, texts_a, texts_b in all_comparisons:
            # Calculate precision, recall, and F1 scores using BERTScore
            P, R, F1 = self.bert_scorer.score(texts_a, texts_b)
            
            # Calculate average scores for the current comparison to summarize over all instances
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()

            # Log the calculated metrics for monitoring and analysis
            self.log(f'{label}_precision', avg_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{label}_recall', avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{label}_f1', avg_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(f"{label}: Precision={avg_precision}, Recall={avg_recall}, F1={avg_f1}")
            

    
    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BART-based similarity scores for a variety of text comparisons,
        using the BARTScorer to evaluate the similarity between different segments of texts.
        """
        print("-- Calculating BART similarity scores --")

        # Define all pairs of text segments for which to calculate similarity scores
        all_comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Iterate over each pair and calculate BARTScores
        for label, src_texts, tgt_texts in all_comparisons:
            # Ensure src_texts is a list of texts and tgt_texts could be a list of lists (for multiple references)
            if isinstance(tgt_texts[0], list):
                # Calculate scores with multi-reference support
                scores = self.bart_scorer.multi_ref_score(src_texts, tgt_texts, agg='mean', batch_size=4)
            else:
                # Calculate scores for single reference
                scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)
            
            # Calculate the average score for simplicity; you might want to log or analyze scores further
            avg_score = sum(scores) / len(scores) if scores else float('nan')

            # Log the average BARTScore using whatever logging mechanism (e.g., self.log in PyTorch Lightning)
            print(f"{label}: {avg_score}")
            self.log(f'{label}_avg_score', avg_score, on_step=False, on_epoch=True, prog_bar=True)


            
    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, calculate the loss, and optionally generate text.
        """
    
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
        """
        print("-- configure_optimizers --") 
        
        #lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        #return torch.optim.AdamW(self.model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def generate_text(self, input_ids, attention_mask, max_length=250):
        """
        Generates text sequences from the provided input components using the model,
        with a customizable maximum length for the generated text.
        """
        print("-- generate_text --")

        # Use the `max_length` argument in the model's generate function to control the maximum length of the generated sequences.
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length  # This is now an adjustable parameter.
        )

        # Decode the generated token IDs back into human-readable text,
        # skipping special tokens to improve readability.
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]

        return generated_texts

