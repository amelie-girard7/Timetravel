# TimeTravel Project: Counterfactual Story Rewriting Evaluation



## Supervised Learning for Story Rewriting

The goal is to fine-tune a pre-trained model (Flan-T5) for the task of counterfactual story rewriting. This involves training the model to generate a story ending that aligns with both the original premise and an introduced counterfactual element.

### Model Input and Output Components

- $p_{\theta}$: The probability distribution parameterized by $(\theta$).
- $s'_{3:5}$: The sequence representing the edited ending.
- $S$: The complete story (x1x2y).
- $[s]$: Separator token.
- $s1$: The premise (x1).
- $s'_{2}$: The counterfactual input (xx2).


```math
L_s(\theta) = \log p_{\theta}(s'_{3:5} \mid S, <s>, s_1, s'_{2})
```
Input: x1x2yx1xx2 (premise, initial, original ending, counterfactual,<s>)
Target output: s'_{3:5}

### Training and Evaluation

The model undergoes training to learn the patterns in the data, guided by the loss function. The evaluation process assesses the quality of the rewritten stories, ensuring they are coherent and consistent with the counterfactual premise.
For a more detailed breakdown of the code structure and how each component contributes to the project, please refer to the `README.md` in the root directory.
