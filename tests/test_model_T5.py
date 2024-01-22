# tests/test_model_T5.py

import unittest
from src.models.model_T5 import FlanT5FineTuner
from transformers import T5Tokenizer
import torch

class TestFlanT5FineTuner(unittest.TestCase):
    
    def setUp(self):
        # Initialize things you'll need for the tests (like model, tokenizer)
        # Note: Use a dummy model or a very small version for testing purposes
        self.model = FlanT5FineTuner('google/t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    def test_forward_pass(self):
        # Test the forward pass of the model
        input_text = "Translate English to French: She likes grapes."
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Perform forward pass
        output = self.model(input_ids=input_ids)
        
        # Check if output is not None
        self.assertIsNotNone(output)
        
        # Check if output has 'logits' key
        self.assertIn('logits', output)
        
        # Check if logits is a tensor
        self.assertIsInstance(output.logits, torch.Tensor)
    
    def test_preprocessing(self):
        # Test your preprocessing functions
        # Assuming you have a function preprocess_data in your FlanT5FineTuner class
        input_data = {
            'premise': "She likes grapes.",
            'initial': "She went to the store.",
            'original_ending': "She bought some grapes.",
            'counterfactual': "She went to the vineyard.",
            'edited_ending': ["She picked some grapes."]
        }
        processed = self.model.preprocess_data(input_data)
        
        # Check if processed data has 'input_ids' and 'output_ids'
        self.assertIn('input_ids', processed)
        self.assertIn('output_ids', processed)
        
        # Check if 'input_ids' and 'output_ids' are not empty
        self.assertTrue(processed['input_ids'])
        self.assertTrue(processed['output_ids'])

    # Add more tests as per your requirement

if __name__ == '__main__':
    unittest.main()
