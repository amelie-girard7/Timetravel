# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attentions, x_tokens, y_tokens, layer_num=0, head_num=0, figsize=(10, 8), example_index=0):
    """
    Visualizes attention weights for a given layer and head for a specific example index.
    
    Parameters:
        attentions (tensor): Attention weights of shape [num_layers, num_heads, seq_len, seq_len].
        x_tokens (list of str): Tokens for the x-axis (generated text).
        y_tokens (list of str): Tokens for the y-axis (original ending).
        layer_num, head_num, figsize, example_index: as previously defined.
    """
    attention = attentions[layer_num][head_num][example_index].detach().cpu().numpy()
    plt.figure(figsize=figsize)
    sns.heatmap(attention, annot=False, cmap='viridis', xticklabels=x_tokens, yticklabels=y_tokens)
    plt.title(f'Attention Map - Layer {layer_num}, Head {head_num}, Example {example_index}')
    plt.xlabel('Generated Text')
    plt.ylabel('Original Ending')
    plt.show()
