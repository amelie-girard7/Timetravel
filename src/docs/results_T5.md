# Objective

The primary aim of this study is to measure the T5 model's efficacy in a story rewriting task, focusing on performance measurement through established evaluation benchmarks such as Sacre BLEU, Sacre ROUGE, and context-aware metrics like BERT and BART. This evaluation utilises a dataset comprising stories with modified conclusions (edited_endings) based on a counterfactual(hypothetical) scenario. The objective is to gauge the model's ability to modify the original story ending to reflect the counterfactual scenario within a supervised learning framework.

We are in the context of supervised learning and this is the daaset. 

# Dataset
- **train_supervised_small.json**: Contains 16,752 examples featuring human-modified endings.
- **dev_data.json**: Comprises 1,871 examples with human-modified endings.
- **test_data.json**: Includes 1,871 examples with human-modified endings.

Dataset Structure:
- **Premise**: Initial context of the story (e.g., "Andrea wanted a picture of her jumping.").
- **Initial**: Event leading up to the counterfactual alteration (e.g., "She set the camera up.").
- **Counterfactual**: Hypothetical scenario introduced (e.g., "She asked her friend to draw one.").
- **Original Ending**: The story's endings without the counterfactual modification (e.g., "Then, she jumped in the air...").
- **Edited Ending**: The altered original endings done by the humans accounting for the counterfactual scenario (e.g., "Then, she jumped in the air to demonstrate...").

#### Example

```json
{
  "story_id": "4fd7d150-b080-4fb1-a592-8c27fa6e1fc8",
  "premise": "Andrea wanted a picture of her jumping.",
  "initial": "She set the camera up.",
  "counterfactual": "She asked her friend to draw one.",
  "original_ending": "Then, she jumped in the air. The picture kept coming out wrong. It took twenty tries to get it right.",
  "edited_ending": [
    "Then, she jumped in the air to demonstrate how she wanted it to look.",
    "The picture kept coming out wrong.",
    "It took drawing it several times to get it right."
  ]
}
```

# Experiment Environment:

**Flan-T5**
For model documentation: https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5

| Model                  | Parameters             | Download Size |
|------------------------|------------------------|---------------|
| google/flan-t5-base    | Roughly 247M           |               |
| google/flan-t5-large   | About 770M             | 1GB           |     

**Metrics**
- BERT Score Documentation: https://github.com/Tiiiger/bert_score (Model: microsoft/deberta-xlarge-mnli)
- BART Score Documentation: https://github.com/neulab/BARTScore (Model: facebook/bart-large-cnn)
- ROUGE Metric Info: https://github.com/pltrdy/rouge
- BLEU Score Info: https://github.com/mjpost/sacrebleu

| Metric      | Range    | Desired Outcome |
|-------------|----------|-----------------|
| Sacre-Bleu  | [0-100]  | High            |
| Sacre-Rouge | [0-1]    | High            |
| BART        | [-Inf, 0]| Higher          |
| BERT        | [0-1]    | High            |


# Expected outcome

To create a comprehensive table including the column for interpretation as you requested, let's compile the expectations and interpretations for each similarity comparison between different elements of the text (such as prediction, edited_ending, counterfactual, initial, original_ending, and premise). This table will help clarify the expectations from the model's outputs in relation to the provided inputs and the edited endings.


|  Similarity                      | Expected Outcome |Interpretation  
|----------------------------------|------------------|---------------------------------------------------------------------------------------|
| (prediction, edited_ending)      |High              | We expect a high similarity as the model's predictions should closely match the edited endings, indicating accurate reflection of the counterfactual in the rewrite. |
| (prediction, counterfactual)     | High             | High similarity is expected since the model's predictions should incorporate elements or concepts from the counterfactual scenario.                                 |
| (prediction, initial)            | Low              | Low similarity is desirable, indicating that the model moves beyond the initial setup to explore new narrative directions.                                         |
| (prediction, original_ending)    | Low (probably)   | A low similarity suggests that the model effectively generates new endings distinct from the original, although some overlap may occur due to narrative constraints.|
| (prediction, Premise)            | High             | High similarity is expected because both the model's predictions and the premise should share thematic or narrative consistency.                                    |
| (edited_ending, counterfactual)  | High             | High similarity is expected as the edited endings are designed to reflect the introduced counterfactual scenario directly.                                          |
| (edited_ending, initial)         | Low              | Low similarity is desirable, indicating that the edited endings diverge significantly from the initial events to accommodate the counterfactual.                    |
| (edited_ending, original_ending) | Low              | A low similarity indicates successful diversion from the original ending due to the integration of the counterfactual, despite potential narrative overlap.         |


------------------------------------------------------------------------------------------------------

## Enhanced Strategy for Adapting Story Endings to Counterfactual Scenarios

Tackling the prediction issue of the model duplicating the original ending instead of adjusting the original ending to suit the counterfactual scenario, this approach aims to enhance the model's ability to modify story endings based on counterfactual situations. By thorough preprocessing, careful modification of inputs, specific weighting of tokens, and the use of an adaptable loss function, the model is equipped to grasp the subtleties present in the training data, leading to the production of story endings that are contextually consistent and conform to the outlined counterfactual scenarios with minimum changes to the original endings.

### 1. Dataset Manipulation: Advanced NLP Techniques for Precise Difference Marking

**Objective**: Improve the model's ability to recognize and adapt to differences between original and edited story endings by using advanced NLP techniques.

#### Preprocessing Steps:
- **Token-level Difference Identification**: Utilize `nltk` and consider integrating `spaCy` for a more nuanced comparison of original and edited endings at the token level. Mark the identified differences with special tokens or flags, aiding model recognition.

- **Marking the Edited Ending (Experiment a)**: By highlighting the edited alterations, this method simplifies the learning curve by making the alteration patterns more evident to the model. The words that are in the edited endings but not in the original endings are marked as <diff>word</diff>


### 2. Model Input Modification to Explicitly Highlight Differences

**Modified Input Structure Example**:
```python
 # Construct the input sequence with all components separated by the T5 eos token
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']}"
            f"{row['original_ending']}{separator_token}"
            f"{row['premise']}{row['counterfactual']}{differences_cleaned}"
        )
```


### 3. Custom Token Weighting within the Model's Architecture



### 4. Loss Function Adjustment for Focused Learning on Counterfactual Adaptations

**Custom Loss Function**:
```python
def custom_loss(output, target, diff_weights, non_diff_weights):
    # Apply differential weighting to the loss calculation to prioritize accuracy in marked segments
    loss = ...  # Implement logic for calculating loss, differentiating between marked and unmarked tokens
    return loss
```
Develop a loss function that differentially applies weights to errors based on whether they occur in marked segments, encouraging the model to focus on adapting the ending accurately.

### 5. Training Procedure Emphasizing Counterfactual Modifications

During training, provide the model with inputs that clearly indicate required narrative changes. Utilize the custom loss function to backpropagate errors, with an emphasis on learning counterfactual modifications accurately, ensuring the model not only recognizes but prioritizes the segments that need altering.



------------------------

- **Marking the Original Ending (Experiment b)**: Direct the model's focus towards segments in the original ending that require adjustments to incorporate the counterfactual scenario. This approach aids in teaching the model about the context and magnitude of necessary modifications.

/data/agirard/Projects/Timetravel/data/transformed/b-marking_original_endings.py

  
- **Marking Both (Experiment c)**: Offer comprehensive guidance by indicating the elements that require change in the original narrative and the nature of these changes, considering the complexity and the benefits of contextual completeness.

/data/agirard/Projects/Timetravel/data/transformed/c-marking_both_endings.py
-----------------------------------------




#### Original paper Results (benchmark)

| Model       | BLEU-4 | ROUGE-L | BERT |
|-------------|--------|---------|------|
| GPT + Sup   | 80.09  | 75.03   | 64.15|
| GPT2-S + Sup| 79.03  | 73.31   | 64.14|
| GPT2-M + Sup| 76.63  | 74.42   | 64.06|


## Experiments 26/03 

### Mask the original ending

((model_2024-03-26-14), (model_2024-03-26-16))

In order to evaluate the model's capability to generate an edited ending without being directly influenced by the original ending, we modified the input sequence construction within the `preprocess_data` function. Essentially,we removed or masked the portion of the input that includes the original ending when constructing `input_sequence`. 

```python
def preprocess_data(row, tokenizer, CONFIG):
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']} {separator_token} "  # Mask the original ending by not including it here.
            f"{row['premise']} {row['counterfactual']}"
        )
```
This adjustment ensures that the model's input sequence does not include the original ending, thereby "masking" it during the model's attempt to generate an edited ending based on the premise, initial event, and counterfactual scenario. This approach helps in assessing the model's ability to creatively alter the ending without leaning on the original ending's content.

### Data augmentation

Augmenting specifically the words in the edited ending that are different from the original ending is a targeted and strategic approach to data augmentation. This technique directly addresses the challenge of the model's tendency to reproduce the original endings by emphasizing the modifications in the edited endings, thereby encouraging the model to recognize and generate the necessary variations. 

Our objective here (1) compare the differences between `original_ending` and `edited_ending`, (2) assign higher weights to the words that differ, and (3) provide these weights to the T5 model to discourage it from copying the original ending,this will involve creating a custom loss function that can penalize the model more for copying the original ending. 

However, directly incorporating word-specific weights into the T5 model training in a way that influences the model to pay attention to certain words over others, especially in a generative task, isn't straightforward due to how the model architecture and training process are designed.

A pragmatic approach involves emphasizing the divergent aspects of the story endings in the data preparation phase, thereby making the model implicitly learn the importance of these differences through the standard training process. Here's how we can align our preprocessing and model training code with this strategy:

### Step 1: Enhance Data Preprocessing
Modify the `preprocess_data` function to include a differential focus on divergent words in `edited_ending` by augmenting or emphasizing these words before tokenization.

```python
# Assuming this function is defined to augment words that differ between original_ending and edited_ending
def augment_differences(original_ending, edited_ending):
    # Your augmentation logic here
    # For simplicity, let's say it returns an augmented edited_ending
    return augmented_edited_ending

def preprocess_data(row, tokenizer):
    # Existing preprocessing steps
    
    # Augment the edited_ending based on differences with original_ending
    augmented_edited_ending = augment_differences(row['original_ending'], ' '.join(row['edited_ending']))
    
    # Proceed with tokenization as before, but use augmented_edited_ending
    tokenized_ending = tokenizer.encode_plus(
        augmented_edited_ending, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
    )
    
    # The rest of the function remains unchanged
```

### Step 2: Implement Augmentation Logic
You'll need a function that identifies the differences between `original_ending` and `edited_ending` and then augments these differences. The augmentation could be as simple as synonym replacement or more complex like generating alternative sentences that convey the same differences but in varied words or structures.

### Step 3: Integrate with Model Training
Ensure your model training process remains unchanged; the key is that the model is now trained on data that inherently emphasizes the differences between the story endings, teaching it the importance of these divergences.

