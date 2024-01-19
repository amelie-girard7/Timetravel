# Dataset and code for "Paper name"

This repo contains the dataset and code for the following paper:

[paper name]()  
*author1, author2, author3*  


Pearl's causal ladder segments problems in data-driven AI into three tiers: observation (“seeing”), intervention (“doing”), and counterfactuals (“imagining”) [1].

**observation***: pertains to the observation of statistical correlations, asking questions like, "How frequently do I take aspirin when I have a headache?" This level focuses on the statistical dependencies between random variables, involving probabilistic reasoning about joint and conditional distributions, symbolized as P(X = x, Y = y) and P(Y = y|X = x). These relationships can be structured using Bayesian Networks [12, 54], which represent a set of variables and their conditional dependencies in a directed acyclic graph (DAG).

**intervention**: involves the formalization of active interventions in the world to achieve a desired outcome, such as, "Will taking an aspirin now relieve my headache?" This level uses the concept of the do-operator [24] and Causal Bayesian Networks [63] to express interventions, for instance, the distribution of Y when X is altered to a specific value x, represented as P(Y = y|do(X = x)).

**counterfactuals**: deals with counterfactual reasoning. This involves contemplating alternative scenarios that might have unfolded differently from reality, even in ways that contradict what actually happened, such as, "Would my headache have gone away if I had taken an aspirin?" Counterfactual probabilities are denoted as P(Yx = y), indicating the likelihood that "Y would be y, had X been x." Addressing the counterfactual concepts necessitates the use of Structural Causal Models (SCMs) [63]. SCMs are particularly potent as they allow for precise formulation of any concept across Rungs 1, 2, and 3 [3].


Progressing through these hierarchy, the complexity of the problem intensifies, demanding an in-depth understanding of causality that extends beyond observed data. This structure brings forth unique challenges and opportunities, especially in how it relates to explainability and intersects with causal studies. Our focus is on natural language processing, where grasping internal causality is essential. This understanding is pivotal for pinpointing and replacing components of models with coded modules, which can improve their reliability and potentially enhance their performance.


# Counterfactual reasoning  (Example from the paper 'Background')
Counterfactual reasoning involves considering alternative  scenarios that deviate from the existing narrative. This concept is widely explored in various fields, including psychology as highlighted by Epstude and Roese (2008), cognitive science as noted by Byrne (2002), and in natural language processing as discussed in works by Hobbs (2005), Lawrence and Riezler (2018), and Son et al. (2017). Despite advancements in Natural Language processing (NLP) through pre-trained language models like BERT (Devlin et al., 2018) and GPT (Radford et al., 2018), these models often struggle to differentiate between plausible and implausible counterfactuals, a challenge outlined in Zellers et al. (2019). Furthermore, when models succeed in tasks involving the discernment of reasonable alternatives, they sometimes rely on hidden biases within the dataset, as observed in studies by Niven and Kao (2019) and Zellers et al. (2018), rather than developing a robust understanding of counterfactual reasoning.
Training models to generate outcomes from counterfactual prompts could foster a deeper comprehension of the underlying situation dynamics, in contrast to merely distinguishing between two alternatives, which might exploit dataset biases (Lianhui et al., 2019) . This approach is akin to script learning, which involves standardizing typical event sequences to comprehend causal narrative structures, as investigated by Pichotta and Mooney (2014) and Chambers (2013). However, capturing the complexity of causal relationships in templated formats is challenging, as indicated by Sap et al. (2019). Therefore, we focus on counterfactual reasoning within unstructured text, requiring models not only to understand but also to generate the outcomes of such reasoning.

 In our tasks, the "counterfactual event" is akin to a causal intervention in the story's event sequence, as conceptualized by Pearl (2000). This demands narrative alterations to align with the common knowledge of how the world operates, thus integrating causal reasoning in a manner accessible to those unfamiliar with formal causality concepts. This framework also helps us assess the strengths and weaknesses of recent developments in neural language models in counterfactual reasoning. counterfactual rewriting delves into the causes and effects within a story, which may necessitate nuanced and varied adjustments to align with the counterfactual event.



# Repository structure 

```
TIMETRAVEL/
│
├── README.md            # Project overview and instructions
├── LICENSE              # License information
├── .gitignore           # Files and folders to be ignored by git
│
├── src/                 # Source code for the project
│   ├── main.py          # Main script to run the models
│   ├── models/          # Model definitions
│   │   └── model_x.py   # Specific model file
│   ├── utils/           # Utility scripts and functions
│   │   └── helper.py    # Helper functions
│   └── data_loader.py   # Data loading and preprocessing scripts
│
├── data/                # Data directory (could be ignored by git if data is large)
│   ├── raw/             # Raw data, unprocessed
│   ├── processed/       # Processed data ready for model input
│   └── external/        # Any external data sources
│
├── notebooks/           # Jupyter notebooks for experiments and analysis
│   └── exploration.ipynb
│
├── requirements.txt     # Project dependencies
│
├── tests/               # Test scripts
│   └── test_model_x.py  # Test for specific model
│
├── scripts/             # Utility scripts, e.g., setup or install scripts
│   └── setup.sh
│
├── models/              # Trained model files (can be ignored by git for large models)
│   └── model_x.pkl
│
└── docs/                # Documentation files
    └── model_x.md       # Documentation specific to a model
```



## Dataset: TimeTravel

The dataset can be **downloaded** from [here](https://drive.google.com/file/d/150jP5FEHqJD3TmTO_8VGdgqBftTDKn4w/view?usp=sharing). 

**Data File Descriptions** (refer to the examples provided):
1. `train_supervised_small.json`: This is the supervised training set utilized in the paper's experiments.
2. `train_supervised_large.json`: A more extensive supervised training set, developed through additional annotations.
3. `train_unsupervised.json`: Set for unsupervised training.
4. `dev_data.json`: Development (Dev) set.
5. `test_data.json`: Set for testing purposes.

**variables**

**s1** -> x1: premise
**s2** -> x2: initial
**s_3:5** -> y: original ending (called "s_3:5" in the paper; )

Threrefore uppercase "S" is x1x2y 

**s'2** -> xx2: counterfactual


**s'_3:5** -> yy: edited ending

Example: 


## Supervised Training Approach (Sup)

In the  dataset, there are 16,752 instances for training that include **rewritten endings created by humans**, offering a foundation for supervised learning. To determine the benefits of training models directly with these alternative endings for a better grasp of counterfactual narratives, we adopt a supervised learning strategy using this data subset. 
In this approach, the model is fed with complete information (S, [s], s1, s′2), 


aiming to maximize the log-likelihood of the actual rewritten endings: 
\( L_s(\theta) = \log p_{\theta}(s'_{3:5}|S, [s], s1, s'_{2}) \) (5),

- \(p_{\theta}\): This represents a probability distribution parameterized by \(\theta\).
- \(s'_{3:5}\): This denotes a sequence or variable that spans from the 3rd element to the 5th element. It represents the edited ending (yy) or a specific part of it.
- \(S\): This stands for the complete story S: x1x2y.
- \([s]\):  signifies a separator token.
- \(s1\): This corresponds to the premise.
- \(s'_{2}\): This represents the counterfactual input.

So, the expression is indicating the conditional probability of observing the sequence \(s'_{3:5}\) ( the edited ending) given the complete information \(S\), a specific token \([s]\), the premise \(s1\), and the counterfactual input \(s'_{2}\), with this probability being parameterized by \(\theta\).


we will  build variables x1x2yx1xx2_ids (input for the model) and yy_ids (output) from the JSON file.



### Step1: Dataset Preparation:

We will Use the 16,752 training instances with human-annotated rewritten endings.

C:\Users\24419222_admin\Documents\GitHub\Timetravel\data\raw\train_supervised_small1.json

***data example***

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


1. **Load the JSON file**: We will start by loading the JSON file (`train_supervised_small1.json`)

The data structure of the JSON file consists of the following fields:

- `story_id`: A unique identifier for each story.
- `premise`: The premise of the story.
- `initial`: The initial ending of the story.
- `counterfactual`: The counterfactual modification of the `initial` story.
- `original_ending`: The original ending of the story.
- `edited_ending`: A one of human-annotated rewritten ending, representing the counterfactual narratives for the supervised traing. For  the validation and test sets have, respectively, 3 and 4 per sample.

### Preprocessing the Data
Based on the structure, we will preprocess the data to create the required format for model training. The input for the model should be a combination of the `premise`, `initial`, `original_ending` and `counterfactual` fields. The output (target) will be the `edited_ending`.

1. **Input Text**: A concatenation of `premise`, `initial`, `original_ending` and `counterfactual` fields.
2. **Output Text (yy_ids)**: The `edited_ending` field.

We will also include visuals to ensure that the data is being processed correctly.

Let's define the preprocessing function and apply it to the dataset.

```python
import pandas as pd

# Load the JSON file to inspect its structure
file_path = 'C:\Users\24419222_admin\Documents\GitHub\Timetravel\data\raw\train_supervised_small1.json'
data = pd.read_json(file_path, lines=True)

# Display the first few rows of the dataset
data.head()
```

### Next Steps:
1. **Verify Processed Data**: Ensure that the processed data aligns with the expected format for the model. We've concatenated the premise, initial, and counterfactual fields, separated by a special token `[s]`, to form the input, and used the first `edited_ending` as the output.
2. **Model Training Preparation**: With the dataset prepared, the next steps involve setting up the training environment, defining the model architecture, and training the model using the processed data.

Would you like to proceed with inspecting a few processed data samples to ensure correctness or move directly into setting up the model training environment?



2. **Training Objective**: Implement the training objective to maximize the log-likelihood of the rewritten endings (Equation 5).
3. **Model Training**: Train the model using the complete information (S, [s], s1, s′2) as input.
4. **Evaluation**: Test the model's performance on generating counterfactual narratives that align with human-created rewritten endings.










































































































































































## Unsupervised Training

### Zero-shot (ZS)
In our most basic scenario, we assess the capability of the model to engage in counterfactual reasoning, a skill acquired through their pretraining on extensive datasets. In this particular context, the model undergoes no training using any part of TIMETRAVEL's training data. Instead, it is required to create counterfactual rewritten narratives for the evaluation set, relying solely on the knowledge gained from its pretraining. During testing, the model is presented with both the premise and the altered counterfactual context (s1, s′2) and is tasked with generating the tokens that form the revised counterfactual result.

**Model Selection**: Choose a pre-trained model, such as Flan-T5, which has been trained on extensive datasets.
**Input Preparation**: For the testing phase, prepare inputs consisting of the original premise and the counterfactual context (s1, s′2). (Todo: test_data.jason : we need to get rid of the endings in the test data)
**Generation**: Use the model to generate tokens forming the revised counterfactual result without any additional training.

* Dev / test data Format
```json
{
  "story_id": "048f5a77-7c17-4071-8b0b-b8e43087132d",
  "premise": "Neil was visiting Limerick in Ireland.",
  "initial": "There, he saw a beautiful sight.",
  "counterfactual": "It was the ugliest city he's ever seen.",
  "original_ending": "He saw the large and lovely River Shannon! After a few minutes, he agreed with the locals. The River Shannon was beautiful.",
  "edited_endings": [
    [
      "He saw the small and lonely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was lonely."
    ],
    [
      "However, he saw the large and lovely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was beautiful."
    ],
    [
      "However, he did think the large River Shannon was lovely!",
      "After a few minutes, he agreed with the locals that Limerick wasn't as ugly as he though.",
      "The River Shannon was beautiful."
    ]
  ]
}
```

```python
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"  # You can choose the appropriate model size
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load your JSON data
test_data = json.loads(your_json_string)

# Prepare input text
prompt = "Rewrite the ending: "
input_text = f"{prompt} premise: {test_data['premise']} counterfactual: {test_data['counterfactual']}"

# Generate rewritten ending
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output_ids = model.generate(input_ids, max_length=200)  # Adjust max_length as needed
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generated_text)
```


### Fine-tuning (FT) 
Given that the training domains for Flan-T5 model is wider-ranging and more intricate than the ROCStories domain, we explore the benefits of aligning the language model more closely with the ROCStories data distribution for enhanced counterfactual reasoning. In this approach, the model undergoes additional fine-tuning to optimize the log likelihood of the narratives in the ROCStories corpus: 
\(Lft(θ) = log pθ(S)\) (1)

where \( pθ \) represents the language model with parameters \( θ \), and \( S \) is the original story. This fine-tuning process is designed to steer the model towards generating text that aligns with the distinctive narrative style of the stories. Mirroring the zero-shot scenario, the model receives the premise and the modified counterfactual sentence (s1, s′2) as inputs.

**Model Selection and Loading**: Start with the same pre-trained model Flan-T5.
**Dataset Preparation**: Prepare the corpus for training (this is a training on the Test data as it we have the endings here).
**Fine-tuning Process**: Fine-tune the model on ROCStories data to maximize the log likelihood of the narratives (Equation 1).
**Input and Evaluation**: Similar to the zero-shot scenario, use the premise and counterfactual sentence as inputs for evaluation.

* Dev / test data example

```json
{
  "story_id": "048f5a77-7c17-4071-8b0b-b8e43087132d",
  "premise": "Neil was visiting Limerick in Ireland.",
  "initial": "There, he saw a beautiful sight.",
  "counterfactual": "It was the ugliest city he's ever seen.",
  "original_ending": "He saw the large and lovely River Shannon! After a few minutes, he agreed with the locals. The River Shannon was beautiful.",
  "edited_endings": [
    [
      "He saw the small and lonely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was lonely."
    ],
    [
      "However, he saw the large and lovely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was beautiful."
    ],
    [
      "However, he did think the large River Shannon was lovely!",
      "After a few minutes, he agreed with the locals that Limerick wasn't as ugly as he though.",
      "The River Shannon was beautiful."
    ]
  ]
}
```

### Fine-tuning + Counterfactual (FT + CF)
The training loss described earlier doesn't utilize the additional 81,407 counterfactual training sentences for fine-tuning. To incorporate a broader array of potential counterfactual narratives from the training data into the model, we introduce an extra loss function. This function tunes the model to align with the counterfactual sentences based on the given premise sentence: 

\( L_{cf} (\theta) = \log p_{\theta}(s'_{2}|s_{1}) \) (2)

where \( p_{\theta}(s'_{2}|s_{1}) \) indicates that the language model first processes the premise \( s_{1} \) and then maximizes the log likelihood of the counterfactual sentence \( s'_{2} \). The model is fine-tuned using both objectives in Eqs (1) and (2): 

\( L_{ft+cf} (\theta) = L_{ft} + L_{cf} \) (3)

, and it receives inputs in the same format as the zero-shot and fine-tuned models during testing.


1. **Extended Training Set**: Incorporate the additional 81,407 counterfactual training sentences.
2. **Additional Loss Function**: Implement the additional loss function (Equation 2) to fit the model to the counterfactual sentences.
3. **Fine-tuning Process**: Fine-tune the model using both the objectives from Equations (1) and (2).
4. **Testing**: Test the model with inputs in the same format as the zero-shot and fine-tuned models.

* Unsupervised training data example 

```json
{
  "story_id": "da0e85f1-c586-4236-a8a3-ee6421c8e71d",
  "premise": "Charles' mother taught her son to carry a pre-paid cell phone.",
  "initial": "As a job seeker, Charles put his cell phone number on applications.",
  "counterfactual": "As a job seeker, Charles used his cell phone to keep his information out of employers hands.",
  "original_ending": "He needed a real cell phone, but kept up with his pre-paid cell phone. One afternoon he was in a phone interview with Apple Computers. He ran out of minutes and never reached Apple's hiring manager again."
}
```
### Reconstruction + Counterfactual (RC + CF)
One limitation of the aforementioned training methods is that they don’t explicitly train models to retain as much of the original story ending \( x_{3:5} \) as possible, aiming for minimal edits. For the models to effectively "rewrite" the original story ending with the counterfactual sentence, rather than creating an entirely new plot, they need to be conditioned on the original ending during generation. Driven by this need and the objective to develop unsupervised methods for counterfactual rewriting, we introduce a reconstruction objective. This objective trains the model to reproduce a noisy version of the original ending. Specifically, we feed the model with both the original story and a masked context as input (S, [s], s1, [mask]) and train it to reconstruct the original ending 

\( s_{3:5} \): \( L_{rc}(\theta) = \log p_{\theta}(s_{3:5}|S, [s], s_{1}, [mask]) \) (4)

where [s] is a separator token and [mask] is a special mask token. 

In this scenario, the model first reads the original story \( S \), followed by the separator token [s], and then revisits the premise \( x_{1} \) and the mask token [mask], which acts as a placeholder for the counterfactual sentence. This training objective encourages the model to recreate the original ending \( s_{3:5} \) in cases where the second sentence is unspecified, promoting generations similar to the original ending regardless of the provided counterfactual.

During testing, we substitute [mask] in the input with the counterfactual sentence \( s'_{2} \), and the model is then required to generate the continuation of (S, [s], s1, \( s'_{2} \)). Additionally, we use the objective from Eq (2) to infuse the model with counterfactual knowledge during training.

1. **Reconstruction Objective**: Implement the reconstruction objective (Equation 4).
2. **Training**: Train the model to reconstruct the original ending from both the original story and a masked context.
3. **Testing**: During testing, replace the mask token with the counterfactual sentence and generate the continuation.

* Unsupervised training data example 

```json
{
  "story_id": "da0e85f1-c586-4236-a8a3-ee6421c8e71d",
  "premise": "Charles' mother taught her son to carry a pre-paid cell phone.",
  "initial": "As a job seeker, Charles put his cell phone number on applications.",
  "counterfactual": "As a job seeker, Charles used his cell phone to keep his information out of employers hands.",
  "original_ending": "He needed a real cell phone, but kept up with his pre-paid cell phone. One afternoon he was in a phone interview with Apple Computers. He ran out of minutes and never reached Apple's hiring manager again."
}
```
C:\Users\24419222_admin\Documents\GitHub\Timetravel\data\raw\train_unsupervised.json




# Rewritten Sentence Scoring Setup

In this experimental setup, workers from Amazon Mechanical Turk evaluated 100 outputs from 14 distinct models. Each worker was presented with four elements for every example: the original premise sentence, the original ending, the counterfactual sentence, and the rewritten ending. They were then asked to rate the following three aspects on a 3-point Likert scale:

1. Does the ***rewritten ending*** maintain details from the original ***premise sentence***?
2. Is the plot of the ***rewritten ending*** relevant to the plot of the ***original ending***?
3. Does the ***rewritten ending*** effectively incorporate the changes induced by the ***counterfactual sentence***?

The following table displays the Likert scale scores ?? How can I replicate it??

| Model             | Pre (1) | Plot (2) | CF (3)  |
|-------------------|---------|----------|---------|
| GPT + ZS          | 1.945   | 1.290    | 1.555   |
| GPT2-S + ZS       | 1.945   | 1.335    | 1.475   |
| GPT2-M + ZS       | 2.435   | 1.615    | 2.045   |
| GPT + FT          | 2.485   | 1.750    | 2.005   |
| GPT2-S + FT       | 2.365   | 1.645    | 1.895   |
| GPT2-M + FT       | 2.580   | 1.790    | 2.070   |

# Automatic Metrics

## Metrics

Overlap Metrics The most commonly employed metrics in text generation assessment involve measuring the textual overlap between a generated sequence and a set of reference sequences from the dataset. 

- **BLEU (Papineni et al., 2002)**: This prominent metric in text generation calculates the overlapping n-grams between generated and reference sequences.
- **ROUGE-L (Lin, 2004)**: Initially designed for extractive summarization, this metric evaluates the longest common subsequence (LCS) between a candidate generation and a reference. We present the performance of all models using these metrics.

#### Model-based Metrics

While BLEU and ROUGE are prevalent, their reliance on exact string matching limits their ability to effectively recognize paraphrases and crucial semantic order changes. 

- **Recent Developments**: There's an increasing focus on developing model-based metrics (Lowe et al., 2017) that utilize trained models and embeddings for sequence evaluation.
- **Word Mover’s Distance (Kusner et al., 2015)**: This metric calculates the distance between two texts as the minimum cost to transform one sequence's word embeddings into the other’s. We use the negative exponential of this distance to derive the Word Mover’s Similarity (WMS).
- **Sentence + Word Mover’s Similarity (S+WMS)**: Introduced by Clark et al. (2019), this method extends WMS to longer texts by incorporating sentence representations alongside word embeddings in its minimum distance calculation.
- **Contextualized Embeddings**: Modern techniques employ contextualized embeddings (Devlin et al., 2018) to assess sequence similarity. We utilize BERTScore (Zhang et al., 2019), which calculates cosine similarity between sentences using BERT encodings. BERTScore has been shown to correlate more strongly with human judgments compared to BLEU, ROUGE, and other learning-based metrics.
- **BERTScore Adaptation**: To tailor BERTScore for our task, we fine-tune BERT on ROCStories following the training framework from Devlin et al. (2018), and compute BERT-FT in the same manner as previously.

### Human Correlation with Metrics

Recent research in text generation (Wiseman et al., 2017) and dialogue (Liu et al., 2016) has highlighted the limitations of automatic metrics in text production tasks. Given the counterfactual rewriting task's semantic depth and the necessity to detect nuanced alterations in event narratives, we expect automatic metrics to struggle with accurately evaluating rewritten endings.

- **Correlation Study**: To examine the alignment between existing evaluation metrics for long-form generation and human perceptions of counterfactual generation quality, we calculated the Pearson Correlation between automatic scores and human evaluations for 800 validation set data points. This includes 300 from gold annotations and 100 each from the 5 GPT2-M variants.
- **Evaluation Method**: For each instance, they employed the same question framework and Likert scale assessment as in §5. The findings are presented in Table 6.

#### Observations:

- **Adherence to Premise and Plot**: The automatic metrics show a reasonable correlation with human scores when it comes to sticking to the premise sentence and the plot.
- **Adherence to the Counterfactual Sentence**: Interestingly, these metrics negatively correlate with question (3) – adherence to the counterfactual sentence. This suggests that they may not effectively measure counterfactual understanding when used in the typical manner (i.e., a higher score indicating better performance).
- **BERTScore Metrics***: The only metrics demonstrating a positive correlation with human scores for counterfactual understanding are the BERTScore metrics. They are suitable for evaluating generations in aspects related to all three questions. However, this correlation is weak, and as indicated in Table 7, there is difficulty in differentiating between models using the BERTScore metrics.









## Citation

```bibtex
@inproceedings{,
    title = "",
    author ="",
    booktitle = "",
    month = "",
    year = "",
    address = "",
    publisher = "",
    url = "",
}
```


Reference 
[1] Pearl, J., & Mackenzie, D. (2019). *The book of why*. Penguin Books.



































