# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attentions, tokens, layer_num=0, head_num=0, figsize=(10, 8), example_index=0):
    """
    Visualizes attention weights for a given layer and head for a specific example index.
    
    Parameters:
        attentions (tuple of tensors): The attention tensors from the model. Structure is typically
                                       [num_layers, num_heads, seq_len, seq_len].
        tokens (list of str): List of tokens corresponding to input IDs used for x and y labels.
        layer_num (int): The layer number of the attention heads to visualize.
        head_num (int): The head number within the specified layer to visualize.
        figsize (tuple): The figure size for the plot.
        example_index (int): Index of the example in the batch for which to visualize attentions.
    """
    # Select the specific layer and head
    attention = attentions[layer_num][head_num][example_index].detach().cpu().numpy()

    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(attention, annot=False, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    plt.title(f'Attention Map - Layer {layer_num}, Head {head_num}, Example {example_index}')
    plt.xlabel('Token from sequence')
    plt.ylabel('Token to sequence')
    plt.show()
