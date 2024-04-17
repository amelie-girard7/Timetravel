# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attentions, tokens, layer_num=0, head_num=0, figsize=(10, 8)):
    """
    Visualizes attention weights for a given layer and head.

    Args:
        attentions: The list of attention tensors from the model. Assumes attentions are already detached and moved to CPU.
        tokens: List of tokens corresponding to the attention weights.
        layer_num: Layer number to visualize.
        head_num: Head number to visualize.
        figsize: Size of the figure.
    """
    # Extract the specific layer and head's attention
    # Note: attentions[layer_num] should be of shape [batch_size, num_heads, seq_length, seq_length]
    attention = attentions[layer_num][head_num].squeeze(0)  # Remove batch dimension if present

    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(attention, annot=False, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    plt.title(f'Layer {layer_num} Head {head_num} Attention Weights')
    plt.xlabel('Tokens in sequence')
    plt.ylabel('Tokens in sequence')
    plt.show()