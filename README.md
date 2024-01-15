Pearl's causal ladder segments problems in data-driven AI into three tiers: observation (“seeing”), intervention (“doing”), and counterfactuals (“imagining”) [1].

**Seeing***: pertains to the observation of statistical correlations, asking questions like, "How frequently do I take aspirin when I have a headache?" This level focuses on the statistical dependencies between random variables, involving probabilistic reasoning about joint and conditional distributions, symbolized as P(X = x, Y = y) and P(Y = y|X = x). These relationships can be structured using Bayesian Networks [12, 54], which represent a set of variables and their conditional dependencies in a directed acyclic graph (DAG).

**Doing**: involves the formalization of active interventions in the world to achieve a desired outcome, such as, "Will taking an aspirin now relieve my headache?" This level uses the concept of the do-operator [24] and Causal Bayesian Networks [63] to express interventions, for instance, the distribution of Y when X is altered to a specific value x, represented as P(Y = y|do(X = x)).

**Imagining**: deals with counterfactual reasoning. This involves contemplating alternative scenarios that might have unfolded differently from reality, even in ways that contradict what actually happened, such as, "Would my headache have gone away if I had taken an aspirin?" Counterfactual probabilities are denoted as P(Yx = y), indicating the likelihood that "Y would be y, had X been x." Addressing Rung 3 concepts necessitates the use of Structural Causal Models (SCMs) [63]. SCMs are particularly potent as they allow for precise formulation of any concept across Rungs 1, 2, and 3 [3].


Progressing through these hierarchy, the complexity of the problem intensifies, demanding an in-depth understanding of causality that extends beyond observed data. This structure brings forth unique challenges and opportunities, especially in how it relates to explainability and intersects with causal studies. Our focus is on natural language processing, where grasping internal causality is essential. This understanding is pivotal for pinpointing and replacing components of models with coded modules, which can improve their reliability and potentially enhance their performance.


# Counterfactual reasoning 
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





Reference 
[1] Pearl, J., & Mackenzie, D. (2019). *The book of why*. Penguin Books.



































