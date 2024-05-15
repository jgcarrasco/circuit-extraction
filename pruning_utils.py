from functools import partial
from einops import einsum, rearrange

import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

#########################
#   CACHING FUNCTIONS   #
#########################

def cache_mean_attn_layer_activations(model, patching_tokens):
    """
    Given a GPT-2 HuggingFace model implementation, gathers the mean activations
    of each attention later when inputting `patching_tokens`.

    Parameters
    ----------
    - `model`: `GPT2LMHeadModel`
    - `patching_tokens`: Tensor of shape `(batch_size, seq_len)`

    Return
    ------
    - `mean_attn_layer_activations`: Tensor of shape `(n_layers, seq_len, d_model)`
    containing the mean activations of the attention layer.
    """
    mean_attn_layer_activations = torch.zeros(model.config.n_layer, patching_tokens.shape[-1], model.config.n_embd)

    def _cache_mean_attn_layer_activations(module, input, output, layer_idx):
        mean_attn_layer_activations[layer_idx] = output[0].mean(0)
        return None

    for layer in range(model.config.n_layer):
        # Gather the activations for that layer
        hook_fn = partial(_cache_mean_attn_layer_activations, layer_idx=layer)
        hook = model.transformer.h[layer].attn.register_forward_hook(hook_fn)
        model(patching_tokens)
        hook.remove()
    return mean_attn_layer_activations


def cache_mean_mlp_activations(model, patching_tokens):
    """
    The exact same function as above, but for MLP layers. The shapes are the same.
    """
    mean_mlp_activations = torch.zeros(model.config.n_layer, patching_tokens.shape[-1], model.config.n_embd)

    def _cache_mean_mlp_activations(module, input, output, layer_idx):
        mean_mlp_activations[layer_idx] = output.mean(0)
        return None

    for layer in range(model.config.n_layer):
        # Gather the activations for that layer
        hook_fn = partial(_cache_mean_mlp_activations, layer_idx=layer)
        hook = model.transformer.h[layer].mlp.register_forward_hook(hook_fn)
        model(patching_tokens)
        hook.remove()
    return mean_mlp_activations


def cache_mean_head_activations(model, patching_tokens):
    """
    The same function as the two above, but for individual attention heads.
    In summary, instead of getting the mean activation for the attention later,
    we separate it into the contribution of each attention head of the layer.
    Check the Appendix of the paper for a deeper explanation.
    """
    d_head = int(model.config.n_embd / model.config.n_head)


    mean_head_activations = torch.zeros(model.config.n_layer, model.config.n_head, patching_tokens.shape[-1], model.config.n_embd)
    attn_layer_biases = torch.zeros(model.config.n_layer, model.config.n_embd)

    def _cache_mean_head_activations(c_proj, input, output, layer_idx):
        h = input[0]
        batch_size, seq_len, d_model = h.size()
        h = h.view(batch_size, seq_len, model.config.n_head, d_head)
        w = c_proj.weight.view(model.config.n_head, d_head, model.config.n_embd)

        h_proj = einsum(
            h, w,
            "batch_size seq_len n_head d_head, n_head d_head d_model -> batch_size seq_len n_head d_model"
        ).mean(0) # seq_len, n_head, d_model
        h_proj = rearrange(h_proj, "seq_len n_head d_model -> n_head seq_len d_model")
        
        mean_head_activations[layer_idx] = h_proj
        attn_layer_biases[layer_idx] = c_proj.bias

    for layer in range(model.config.n_layer):
        hook_fn = partial(_cache_mean_head_activations, layer_idx=layer)
        hook = model.transformer.h[layer].attn.c_proj.register_forward_hook(hook_fn)
        model(patching_tokens)
        hook.remove()

    return mean_head_activations, attn_layer_biases


#################
#   MODULES     #
#################

class BiasLayer(nn.Module):
    """
    This module replaces the GPT2Attention layer and optionally outputs a bias term
    """
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, hidden_states, **kwargs):
        return (self.bias, None)


class BiasLayerMLP(nn.Module):
    """
    This module replaces the GPT2MLP layer and optionally outputs a bias term
    """
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, hidden_states, **kwargs):
        return self.bias


class AddLayer(nn.Module):
    """
    This module replaces the GPT2Attention layer and optionally sums a bias term
    """
    def __init__(self, attn: GPT2Attention, bias):
        super().__init__()
        self.bias = bias
        self.attn = attn

    def forward(self, hidden_states, **kwargs):
        output = self.attn(hidden_states)
        return (output[0] + self.bias, None)


class PassthroughLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        return (hidden_states, None)


#########################
#   PRUNING FUNCTIONS   #
#########################


def get_heads_to_prune(circuit_attn_heads, n_heads=12, n_layers=12):
    """
    Given a list of the attn heads of the circuit, returns
    a dictionary heads_to_prune[layer] = [head, ...] with
    every attention head outside of the circuit.
    """
    heads_to_prune = {}
    for layer in range(n_layers):
        heads_to_prune[layer] = [head for head in range(n_heads)]

    for layer, head in circuit_attn_heads:
        heads_to_prune[layer].remove(head)
        
    return heads_to_prune


def get_attn_layers_to_prune(heads_to_prune, n_heads=12):
    """
    If heads_to_prune[layer] contains every head of the attention layer,
    we directly remove the complete layer instead of every separate head.
    """
    attn_layers_to_prune = []
    for layer in heads_to_prune.keys():
        if len(heads_to_prune[layer]) == n_heads:
            attn_layers_to_prune.append(layer)
    for layer in attn_layers_to_prune:
        del heads_to_prune[layer]
    return heads_to_prune, attn_layers_to_prune


def prune_model(model, circuit_attn_heads, circuit_mlps, patching_tokens, ablation_scheme="mean"):
    
    # Gather the corresponding activations
    mean_head_activations, _ = cache_mean_head_activations(model, patching_tokens)
    mean_attn_layer_activations = cache_mean_attn_layer_activations(model, patching_tokens)
    mean_mlp_activations = cache_mean_mlp_activations(model, patching_tokens)

    # PRUNE ATTENTION
    heads_to_prune = get_heads_to_prune(circuit_attn_heads)
    heads_to_prune, attn_layers_to_prune = get_attn_layers_to_prune(heads_to_prune)

    # Replace complete attn layers by just a bias term
    for layer in attn_layers_to_prune:
        model.transformer.h[layer].ln_1 = PassthroughLayer()
        bias = mean_attn_layer_activations[layer].cuda() 
        if ablation_scheme == "zero":
            bias = torch.zeros_like(bias).cuda()
        model.transformer.h[layer].attn = BiasLayer(bias=bias)

    # Prune the individual heads 
    model.transformer._prune_heads(heads_to_prune)

    # Add the bias term of the pruned heads to the respective attention layers
    for layer in heads_to_prune.keys():
        bias = mean_head_activations[layer, heads_to_prune[layer]].sum(0).cuda()        
        if ablation_scheme == "zero":
            bias = torch.zeros_like(bias).cuda()
        model.transformer.h[layer].attn = AddLayer(
            attn=model.transformer.h[layer].attn,
            bias=bias
            )

    # PRUNE MLP
    mlps_to_prune = [mlp for mlp in range(model.config.n_layer) if mlp not in circuit_mlps]
    # Replace MLPs
    for layer in mlps_to_prune:
        model.transformer.h[layer].ln_2 = PassthroughLayer()
        model.transformer.h[layer].mlp = BiasLayerMLP(bias=mean_mlp_activations[layer].cuda())

    return model