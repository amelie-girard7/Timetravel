# src/models/model_T5.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from sacrebleu.metrics import BLEU
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util as sentence_transformers_util
from transformers import BartModel, BartTokenizer
from torch.nn.functional import cosine_similarity




import logging

from src.utils.config import CONFIG  # Import the CONFIG

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
        
        # Initialise sacre bleu
        self.sacre_bleu = BLEU()      
        # Initialize Rouge
        self.rouge = Rouge()
        # Initialise BERT
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialise BART for similarity metric
        self.bart_model = BartModel.from_pretrained('facebook/bart-large')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        
        # Initialize a list to store outputs for each validation step
        self.current_val_step_outputs = []
    
    
    
    def forward(self, input_ids, attention_mask, labels):
        """
        Performs the forward pass of the model.b4         
        Args:premise (Tensor): Tokenized tensor for the story premises.
            initial (Tensor): Tokenized tensor for the initial states of the stories.
            original_ending (Tensor): Tokenized tensor for the original endings of the stories.
            counterfactual (Tensor): Tokenized tensor for the counterfactual (alternative scenarios) of the stories.
            labels (Tensor, optional): Tokenized tensor for the edited endings, serving as labels for training. Default is None.
            attention_mask (Tensor, optional): Tensor indicating which tokens should be attended to, and which should not.
        
        Returns:
            The output from the T5 model, which includes loss when labels are provided, and logits otherwise.
        """
        print("--forward pass--")
        
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        
        # Pass the concatenated input_ids, attention_mask, and labels (if provided) to the model.
        # The T5 model expects input_ids and attention_mask for processing.
        # If labels are provided (during training), the model will also return the loss
        # which can be used to update the model's weights.
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # If labels were provided, the model's output will include loss for training.
        # During inference (no labels), the model generates logits from which we can derive predictions.
        if labels is not None:
            print("Loss from model output:", output.loss.item())
        else:
            print("Model output generated without calculating loss (inference mode).")
        
        return output



    def training_step(self, batch, batch_idx):
        """  
        Defines the training logic for a single batch, where a forward pass is performed and the loss is calculated.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the batch.
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
        
        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.
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
        print(f"Validation loss: {val_loss.item()}")

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
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
        comparisons = [
            ('bleu_prediction_edited', all_generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, counterfactuals_refs),
            ('bleu_prediction_initial', all_generated_texts, initials_refs),
            ('bleu_prediction_original', all_generated_texts, original_endings_refs),
        ]
        
        # Calculate and log BLEU scores for edited_ending vs. other story components
        edited_comparisons = [
            ('bleu_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bleu_edited_ending_initial', all_edited_endings, all_initials),
            ('bleu_edited_ending_original', all_edited_endings, all_original_endings),
        ]
        
        # Combine all comparisons into a single list for processing
        all_comparisons = comparisons + edited_comparisons

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
        rouge_score_comparisons = [
            ('rouge_prediction_edited', all_generated_texts, all_edited_endings),
            ('rouge_prediction_cf', all_generated_texts, all_counterfactuals),
            ('rouge_prediction_initial', all_generated_texts, all_initials),
            ('rouge_prediction_original', all_generated_texts, all_original_endings),
        ]

        # Additional comparisons for edited endings vs. other story components
        edited_comparisons = [
            ('rouge_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('rouge_edited_ending_initial', all_edited_endings, all_initials),
            ('rouge_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Combine all comparisons into a single list for processing
        all_comparisons = rouge_score_comparisons + edited_comparisons

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
        original_comparisons = [
            ('bert_prediction_edited', all_generated_texts, all_edited_endings),
            ('bert_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bert_prediction_initial', all_generated_texts, all_initials),
            ('bert_prediction_original', all_generated_texts, all_original_endings),
        ]
        edited_comparisons = [
            ('bert_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bert_edited_ending_initial', all_edited_endings, all_initials),
            ('bert_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Combine all comparisons into a single list for processing
        all_comparisons = original_comparisons + edited_comparisons

        # Calculate and log BERT similarity scores for each comparison
        for label, texts_a, texts_b in all_comparisons:
            embeddings_a = self.sentence_transformer.encode(texts_a, convert_to_tensor=True)
            embeddings_b = self.sentence_transformer.encode(texts_b, convert_to_tensor=True)
            
            # Compute cosine similarities for each pair of texts
            cosine_scores = sentence_transformers_util.cos_sim(embeddings_a, embeddings_b)
            
            # For each text pair, find the highest similarity with any reference
            max_similarities = cosine_scores.max(dim=1).values
            
            # Log average of the maximum similarity scores
            avg_similarity = max_similarities.mean().item()
            self.log(label, avg_similarity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(f"{label}: {avg_similarity}")

    
    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_original_endings):
        """
        Calculates and logs BART-based similarity scores.
        """
        print("Calculating BART similarity scores...")
        
        # Assuming you have a method to generate embeddings with BART
        # This is a simplified placeholder showing the concept
        def get_bart_embeddings(texts):
            encoded_input = self.bart_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                model_output = self.bart_model(**encoded_input)
            embeddings = model_output.last_hidden_state.mean(dim=1)  # Mean pooling
            return embeddings

        # Define comparisons similar to BERT
        comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
        ]
        
        edited_comparisons = [
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Combine all comparisons into a single list for processing
        all_comparisons = comparisons + edited_comparisons

        # Calculate and log similarity scores for each comparison
        for label, texts_a, texts_b in all_comparisons:
            embeddings_a = get_bart_embeddings(texts_a)
            embeddings_b = get_bart_embeddings(texts_b)

            # Calculate cosine similarity and log the average similarity score
            for i, embedding_a in enumerate(embeddings_a):
                similarities = [cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)).item() for embedding_b in embeddings_b]
                avg_similarity = sum(similarities) / len(similarities)
                self.log(f'{label}_{i}', avg_similarity, on_step=False, on_epoch=True, prog_bar=True, logger=True)


            
    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, calculate the loss, and optionally generate text.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: Output dictionary containing generated texts and metrics.
        """
    
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
            
        Returns:
            The optimizer to be used for training the model.
        """
        print("-- configure_optimizers --") 
        
        #lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        #return torch.optim.AdamW(self.model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def generate_text(self, input_ids, attention_mask):
        """
        Generates text sequences from the provided input components using the model.
        """
        print("-- generate_text --") 

        # Generate a tensor of token IDs based on the input_ids and attention_mask
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Decode the generated token IDs back into human-readable text
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts