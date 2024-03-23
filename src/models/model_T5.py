# src/models/model_T5.py
import csv
import logging
import os
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
from src.BARTScore_metric.bart_score import BARTScorer
from src.utils.config import CONFIG

bart_score_path = str(CONFIG["bart_score_dir"])
if bart_score_path not in sys.path:
    sys.path.append(bart_score_path)

logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model on a specific dataset.
    Incorporates evaluation metrics such as BLEU, ROUGE, BERTScore, and BARTScore for performance metrics.
    """

    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer, along with metric evaluators.

        Args:
            model_name (str): The name of the T5 model to be used.
            model_dir (str): Directory path for saving model-related files.
        """
        super().__init__()
        
        # Loading the T5 model and tokenizer.
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
         # Use the same timestamped model_dir for CSVs and model checkpoints
        self.val_csv_file_path = model_dir / "validation_details.csv"
        self.test_csv_file_path = model_dir / "test_details.csv"

        
        # Initializing metrics for validation.
        self.sacre_bleu = BLEU()      
        self.rouge = Rouge()
        
        
         # Updated BERTScorer and BARTScorer initialization
        self.bert_scorer = BERTScorer(
            model_type=CONFIG["bert_scorer_model_type"],
            device=CONFIG["scorer_device"],
            num_layers=None,  # Consider making this configurable if necessary
            batch_size=CONFIG["bert_scorer_batch_size"]
        )
        self.bart_scorer = BARTScorer(
            device=CONFIG["scorer_device"],
            checkpoint=CONFIG["bart_scorer_checkpoint"]
        )

        # Initialize the list to store validation step outputs for aggregating results over an epoch
        self.current_val_step_outputs = []
        
        # Temporary storage for Text-generation sanity check
        self.epoch_validation_details = []
 
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass of the model. If labels are provided, it returns
        the loss; otherwise, it returns logits.

        Parameters:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor representing attention masks.
            labels (torch.Tensor, optional): Tensor of labels. If provided, the loss is returned.

        Returns:
            ModelOutput: The output from the T5 model. Includes loss if labels are provided.
        """
        # The model handles both inference and training based on whether labels are provided.
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step.

        Args:
            batch (dict): A single batch of data from the DataLoader.
            batch_idx (int): The index of the batch in the dataset.

        Returns:
            torch.Tensor: The loss tensor from the forward pass.
        """   
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
        Executes a validation step, generating predictions and calculating metrics.

        Args:
            batch (dict): A single batch of data from the DataLoader.
            batch_idx (int): The index of the batch in the dataset.
        """
        
        # @Inigo, we don't need this with Lightning
        #self.model.eval()
        
        # Forward pass to compute loss and model outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text
        edited_endings = batch['edited_ending']
        
         # Prepare data for detailed analysis or logging
        validation_details = [{
            'Epoch': self.current_epoch,
            'Premise': premise,
            'Initial': initial,
            'Counterfactual': counterfactual,
            'Original Ending': original_ending,
            'Edited Ending': edited_ending,
            'Generated Text': generated_text
        } for premise, initial, counterfactual, original_ending, edited_ending, generated_text 
        in zip(batch['premise'], batch['initial'], batch['counterfactual'], batch['original_ending'], batch['edited_ending'], generated_texts)]

        self.epoch_validation_details.extend(validation_details)

        # Store output information for metric calculation at the end of the epoch
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            'premises': batch['premise'],
            'counterfactuals': batch['counterfactual'],
            'original_endings': batch['original_ending'],
            'initials': batch['initial'],
            'premises': batch['premise']
        }
        self.current_val_step_outputs.append(output)
        self.log_dict({"avg_val_loss": val_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Example of structured logging

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations to perform at the end of each validation epoch.
        """
        # Aggregate texts from the outputs
        aggregated_texts = self.aggregate_texts()
        # Log calculated metrics
        self.log_metrics(aggregated_texts)
    
        csv_file_path = self.determine_csv_path(test_flag)
        # Check if the file exists
        self.log_to_csv(csv_file_path, self.epoch_validation_details)
        self.cleanup_epoch_data()

    def aggregate_texts(self):
        """
        Aggregates texts from the current validation step outputs for metrics calculation.
        """
        aggregated_texts = {
            'generated': [],
            'edited_endings': [],
            'counterfactuals': [],
            'initials': [],
            'premises': [],
            'original_endings': []
        }

        for output in self.current_val_step_outputs:
            for key in aggregated_texts.keys():
                aggregated_texts[key].extend(output[key])

        return aggregated_texts

    def log_metrics(self, aggregated_texts):
        """
        Logs various similarity scores based on aggregated texts.
        """
        # Adjust this section to correctly unpack aggregated_texts and pass to the metric functions
        try:
            self.calculate_and_log_bleu_scores(
                all_generated_texts=aggregated_texts['generated'],
                all_edited_endings=aggregated_texts['edited_endings'],
                all_counterfactuals=aggregated_texts['counterfactuals'],
                all_initials=aggregated_texts['initials'],
                all_original_endings=aggregated_texts['original_endings']
            )
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
        
        try:
            self.calculate_and_log_rouge_scores(
                all_generated_texts=aggregated_texts['generated'],
                all_edited_endings=aggregated_texts['edited_endings'],
                all_counterfactuals=aggregated_texts['counterfactuals'],
                all_initials=aggregated_texts['initials'],
                all_original_endings=aggregated_texts['original_endings']
            )
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
    
        try:
            self.calculate_and_log_bert_similarity(
                all_generated_texts=aggregated_texts['generated'],
                all_edited_endings=aggregated_texts['edited_endings'],
                all_counterfactuals=aggregated_texts['counterfactuals'],
                all_initials=aggregated_texts['initials'],
                all_original_endings=aggregated_texts['original_endings']
            )
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
        
        try:
            self.calculate_and_log_bart_similarity(
                all_generated_texts=aggregated_texts['generated'],
                all_edited_endings=aggregated_texts['edited_endings'],
                all_counterfactuals=aggregated_texts['counterfactuals'],
                all_initials=aggregated_texts['initials'],
                all_original_endings=aggregated_texts['original_endings']
            )
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")

    def determine_csv_path(self, test_flag):
        return self.test_csv_file_path if test_flag else self.val_csv_file_path

    def log_to_csv(self, csv_file_path, details):
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(details)

    def cleanup_epoch_data(self):
        self.epoch_validation_details.clear()
        self.current_val_step_outputs.clear()

    def calculate_and_log_bleu_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BLEU scores for generated texts against various reference components.
        """
        # Prepare reference lists for BLEU calculations
        edited_endings_refs = [[ending] for ending in all_edited_endings]
        counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        initials_refs = [[init] for init in all_initials]
        original_endings_refs = [[orig] for orig in all_original_endings]

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
                bleu_result = self.sacre_bleu.corpus_score(texts, references)
                bleu_score = bleu_result.score
                self.log(label, bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            except Exception as e:
                print(f"Error calculating {label}: {e}")

    def calculate_and_log_rouge_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs ROUGE scores for the generated texts against various components,
        and also for edited endings against other story components.
        """
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
            for score_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                if score_type in rouge_scores:
                    self.log(f"{label}_{score_type}_f", rouge_scores[score_type]['f'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{label}_{score_type}_p", rouge_scores[score_type]['p'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{label}_{score_type}_r", rouge_scores[score_type]['r'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def calculate_and_log_bert_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BERT similarity scores for generated texts against various components,
        and also for edited endings against other story components.
        """
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
            P, R, F1 = self.bert_scorer.score(texts_a, texts_b)
            # Calculate average scores for the current comparison to summarize over all instances
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()
            self.log(f'{label}_precision', avg_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{label}_recall', avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{label}_f1', avg_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
              
    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BART-based similarity scores for a variety of text comparisons,
        using the BARTScorer to evaluate the similarity between different segments of texts.
        """
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
            if isinstance(tgt_texts[0], list):
                scores = self.bart_scorer.multi_ref_score(src_texts, tgt_texts, agg='mean', batch_size=4)
            else:
                scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)
            # Calculate the average score for simplicity; you might want to log or analyze scores further
            avg_score = sum(scores) / len(scores) if scores else float('nan')
            self.log(f'{label}_avg_score', avg_score, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, 
        calculate the loss, and optionally generate text.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(test_flag=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
        """
        lr = CONFIG["learning_rate"]
        return torch.optim.AdamW(self.parameters(), lr=lr)

    def generate_text(self, input_ids, attention_mask):
        """
        Generates text sequences from the provided input components using the model,
        with a customizable maximum length for the generated text.
        """
        generated_ids = self.model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_length=CONFIG["max_gen_length"]
        )

        #@Inigo do you think that skipping the special token needs to be set to default due to our input?
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]

        return generated_texts

    