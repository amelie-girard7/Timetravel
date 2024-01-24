# TimeTravel Project: Counterfactual Story Rewriting Evaluation

### Objective:

The primary aim of this project is to rigorously evaluate the performance of models specialized in counterfactual story rewriting. The evaluation process begins with employing established metrics such as BLEU and ROUGE, followed by more context-aware metrics like BERT and BART. These metrics provide a quantitative foundation for assessing the model's performance. However, given the inherent complexity and nuanced nature of counterfactual reasoning, the project also seeks to transcend these traditional metrics. It aims to develop and implement a linguistically driven evaluative approach. This approach will focus on analyzing the linguistic differences between the counterfactual and the generated endings and between the initial and the generated endings. The intent is to explore the linguistic transformations and narrative shifts brought forth by the counterfactual intervention, thereby offering a deeper, more nuanced understanding of the model's capabilities in crafting coherent and contextually relevant counterfactual narratives.

### Task Description:

In our tasks, the concept of a "counterfactual event" serves as a pivotal point that triggers alterations within the story's sequence of events. This mirrors the causal interventions as described by Pearl (2000). The introduction of a counterfactual event necessitates narrative modifications that must align with the general, widely accepted understanding of how events unfold in the real world. This process not only demands the integration of causal reasoning in a manner that's comprehensible to individuals unacquainted with complex causality theories but also provides a platform to assess the current capabilities and limitations of neural language models concerning counterfactual reasoning.

Counterfactual rewriting is not merely about altering the narrative; it's about understanding and narratively weaving the intricate web of causes and effects within a story. This task often requires detailed and diverse adjustments in the narrative to ensure that the new trajectory of the story resonates authentically with the introduced counterfactual element. The endeavour is to ensure that these narrative alterations are not just plausible but also retain a strong coherence with the original premise, thereby reflecting a deep and nuanced understanding of the narrative's causal structure.

The evaluation metrics traditionally used, such as BLEU, ROUGE, BERTScore, and adaptations of BART, provide valuable quantitative insights. However, our goal is to augment these insights with a linguistically driven analysis that probes deeper into the narrative alterations induced by counterfactual reasoning. By examining the linguistic nuances and narrative shifts, we aim to offer a more comprehensive, multi-dimensional understanding of the model's performance in counterfactual story rewriting.

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
L_s(\theta) = \log p_{\theta}(s'_{3:5} \mid S, [s], s_1, s'_{2})
```
Input: x1x2yx1xx2 (premise, initial, original ending, counterfactual)
Target output: s'_{3:5} 


### Training and Evaluation

The model undergoes training to learn the patterns in the data, guided by the loss function. The evaluation process assesses the quality of the rewritten stories, ensuring they are coherent and consistent with the counterfactual premise.


For a more detailed breakdown of the code structure and how each component contributes to the project, please refer to the `README.md` in the root directory.
