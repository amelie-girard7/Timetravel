# src/models/model_T5.py
import csv
import logging
import os
import sys
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
from src.BARTScore_metric.bart_score import BARTScorer
from src.utils.config import CONFIG
from bertviz import model_view, head_view
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        """
        super().__init__()

        # Load the configuration for the model with output_attentions enabled
        #config = T5Config.from_pretrained(model_name, output_attentions=CONFIG["output_attentions"])
        #config = T5Config.from_pretrained(model_name, output_attentions=True)
        config = T5Config.from_pretrained(model_name)
        config.output_attentions = True

        # Loading the T5 model and tokenizer.
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        #print("Model initialized with configuration:", config)
        
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
        Performs the forward pass of the model. If labels are provided, it calculates the loss; 
        otherwise, it returns logits. This method also handles the retrieval of attention outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True  # Ensure attentions are returned
        )
        
        # Debug prints for forward outputs
        print("Forward pass outputs:")
        print(f"logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'attentions'):
            print(f"attentions shape: {outputs.attentions[0].shape}")
        else:
            print("No attentions returned")

        return outputs

    def custom_loss(self,outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        
        This function modifies the standard cross-entropy loss by applying a different
        weight to each token based on its importance, which is determined by the differential_weights tensor.
        This is particularly useful for focusing the model's learning on specific parts of the input data.
        """
        # Flatten all tensors to align shapes for element-wise operations
        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten to [batch_size * seq_length]
        # Flatten differential weights to match the sequence tokens
        differential_weights_flat = differential_weights.view(-1)  # Flatten to [batch_size * seq_length]  
        # It's critical to ensure the shapes match up for logits, targets, and differential weights.
        # This check helps avoid potential errors during training.

        print("Custom loss calculation:")
        print(f"logits_flat shape: {logits_flat.shape}, type: {type(logits_flat)}")
        print(f"targets_flat shape: {targets_flat.shape}, type: {type(targets_flat)}")
        print(f"differential_weights_flat shape: {differential_weights_flat.shape}, type: {type(differential_weights_flat)}")

        if logits_flat.size(0) != differential_weights_flat.size(0):
           raise ValueError("The size of logits and differential weights does not match, indicating a potential issue in preprocessing or batch assembly.")
        
        # Compute the standard cross-entropy loss without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat
     
        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()
        
        return mean_weighted_loss

    def training_step(self, batch, batch_idx):
        """
            Executes a training step using either differential weights for the loss calculation
            if configured, or the default loss from the model.

            This method processes a single batch of data, performing a forward pass through
            the model and calculating the loss either using a custom loss function or the default one.
            The custom loss function applies differential weights to each token in the input sequence,
            allowing the model to focus more on certain tokens that are deemed more important according
            to the differential_weights tensor provided in the batch.
        """ 
        print("Training step:")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}, type: {type(batch['input_ids'])}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}, type: {type(batch['attention_mask'])}")
        print(f"Batch labels shape: {batch['labels'].shape}, type: {type(batch['labels'])}")
        if 'differential_weights' in batch:
            print(f"Batch differential_weights shape: {batch['differential_weights'].shape}, type: {type(batch['differential_weights'])}")

        # Perform a forward pass through the model to get the outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Check the 'use_custom_loss' config and whether 'differential_weights' is in the batch
        if CONFIG['use_custom_loss'] and 'differential_weights' in batch:
            # Calculate the loss using the custom loss function, which utilizes differential weights
            loss = self.custom_loss(outputs.logits, batch['labels'], batch['differential_weights'])
        else:
            # Use the default loss provided by the model outputs
            loss = outputs.loss

            # Log the custom calculated loss for monitoring. Logging it as 'train_loss' allows tracking within the Lightning framework.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))


       # Return the loss for backpropagation.
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("Validation step:")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}, type: <class 'torch.Tensor'>")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}, type: <class 'torch.Tensor'>")
        print(f"Batch labels shape: {batch['labels'].shape}, type: <class 'torch.Tensor'>")

        # Perform forward pass
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        # Check attentions in outputs
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"Outputs attentions shape: {outputs.attentions[0].shape}")
        else:
            print("No attentions found in outputs")

        # Calculate validation loss
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

        # Generate text and capture attentions
        generated_texts, self_attentions, cross_attentions = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Debug prints for generated texts
        print("Generated texts:")
        for idx, text in enumerate(generated_texts):
            print(f"Text {idx}: {text}")

        edited_endings = batch['edited_ending']

        # Prepare validation details for logging
        validation_details = [{
            'Epoch': self.current_epoch,
            'Premise': premise,
            'Initial': initial,
            'Counterfactual': counterfactual,
            'Original Ending': original_ending,
            'Edited Ending': edited_ending,
            'Generated Text': generated_text,
        } for premise, initial, counterfactual, original_ending, edited_ending, generated_text
        in zip(batch['premise'], batch['initial'], batch['counterfactual'], batch['original_ending'], batch['edited_ending'], generated_texts)]

        self.epoch_validation_details.extend(validation_details)

        # Collect outputs for this validation step
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            'premises': batch['premise'],
            'counterfactuals': batch['counterfactual'],
            'original_endings': batch['original_ending'],
            'initials': batch['initial'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'outputs': outputs
        }

        self.current_val_step_outputs.append(output)

        # Visualize attention if configured
        if CONFIG["log_attentions"]:
            if batch_idx == 0:
                self.visualize_attention(batch['input_ids'], batch['attention_mask'], batch['input_ids'], cross_attentions, self_attentions)

        # Log average validation loss
        self.log_dict({"avg_val_loss": val_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

    def generate_text(self, input_ids, attention_mask):
        # Enable output of attentions and return dict in generate
        self.model.config.return_dict_in_generate = True
        self.model.config.output_attentions = True

        # Generate text sequences and capture attentions
        generated_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=CONFIG["max_gen_length"],
            output_attentions=True,  # Ensure attentions are returned
            return_dict_in_generate=True  # Return a dictionary including attentions
        )

        # Decode generated sequences into text
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids.sequences
        ]

        # Debug prints for generated texts
        print("Generated text sequences:")
        for idx, text in enumerate(generated_texts):
            print(f"Generated text {idx}: {text}")

        # Capture attentions from the generation process
        cross_attentions = generated_ids.cross_attentions if hasattr(generated_ids, 'cross_attentions') else None
        self_attentions = generated_ids.attentions if hasattr(generated_ids, 'attentions') else None

        # Debug prints for attentions
        if cross_attentions:
            print("Cross attentions shapes:")
            for i, layer_attns in enumerate(cross_attentions):
                print(f"Layer {i} cross attentions shapes:")
                for j, attn in enumerate(layer_attns):
                    print(f"Head {j} shape: {attn.shape}")
        else:
            print("No cross attentions")

        if self_attentions:
            print("Self attentions shapes:")
            for i, layer_attns in enumerate(self_attentions):
                print(f"Layer {i} self attentions shapes:")
                for j, attn in enumerate(layer_attns):
                    print(f"Head {j} shape: {attn.shape}")
        else:
            print("No self attentions")

        # Return generated texts along with self and cross attentions
        return generated_texts, self_attentions, cross_attentions

    def visualize_attention(self, input_ids, attention_mask, labels, cross_attentions, self_attentions):
        # Convert tensors to tokens for visualization
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        label_tokens = self.tokenizer.convert_ids_to_tokens(labels[0])
        
        # Debug prints for tokens
        print(f"Input Tokens: {input_tokens}")
        print(f"Generated Tokens: {label_tokens}")
        
        # Visualize cross-attentions
        if cross_attentions:
            print("Cross-Attention Visualization")
            for i, layer_attns in enumerate(cross_attentions):
                print(f"Layer {i} cross attentions shapes:")
                for j, attn in enumerate(layer_attns):
                    attn = attn.squeeze().mean(dim=1).detach().cpu().numpy()  # Average over heads
                    self.plot_attention(attention=attn, tokens=input_tokens, title=f"Cross-Attention Layer {i}")

        # Visualize self-attentions
        if self_attentions:
            print("Self-Attention Visualization")
            for i, layer_attns in enumerate(self_attentions):
                print(f"Layer {i} self attentions shapes:")
                for j, attn in enumerate(layer_attns):
                    attn = attn.squeeze().mean(dim=1).detach().cpu().numpy()  # Average over heads
                    self.plot_attention(attention=attn, tokens=label_tokens, title=f"Self-Attention Layer {i}")

    def plot_attention(self, attention, tokens, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, ax=ax, cmap="viridis")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")
        ax.set_title(title)
        plt.show()

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations to perform at the end of each validation epoch.
        """
        # Aggregate texts from the outputs for metric calculation
        aggregated_texts = self.aggregate_texts()
        self.log_metrics(aggregated_texts)

        # Handle CSV logging
        csv_file_path = self.determine_csv_path(test_flag)
        if self.epoch_validation_details:  # Check if there are details to log
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")

        # Clean up stored data from the current validation epoch
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

    def normalize_attention(attention):
        # Normalize attention across heads by averaging
        attention = attention.mean(dim=1).detach().cpu().numpy()
        
        # Debug print for normalized attention
        print(f"Normalized attention shape: {attention.shape}")
        
        return attention

    def plot_heatmap(attention, tokens_a, tokens_b):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Debug prints for tokens and attention
        print("Tokens A:", tokens_a)
        print("Tokens B:", tokens_b)
        print("Attention shape:", attention.shape)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention, xticklabels=tokens_b, yticklabels=tokens_a, cmap='viridis', ax=ax)
        plt.xlabel('Generated Tokens')
        plt.ylabel('Input Tokens')
        plt.title('Cross-Attention Heatmap')
        plt.show()


