from functools import partial

import torch.nn.functional as F

from transformer_lens import utils

from utils import kl_div


def ablate_head(activations, hook, head_idx, scheme, new_cache=None):
    """
        Ablates a head from the layer specified by hook and the 
        index specified by `head_idx`.

        Parameters
        ----------
        - `activations`: output of the hook. 
        Usually will have shape `(batch, pos, head, d_head)`
        - `hook`: This specifies where the hook will be located. Usually will be
        of the shape 'blocks.0.attn.hook_result'
        - `head_idx`: The index of the head that we want to ablate
        - `scheme`: Either "zero" or "mean" for zero or mean ablation.
        - `new_cache`: Cache that will be used when performing mean ablation
    """
    assert scheme in ["mean", "zero"], "the ablation scheme should be either 'mean' or 'zero'"

    if scheme == "mean":
        assert new_cache is not None, "`new_cache` is required when mean ablating"
        activations[:, :, head_idx] = new_cache[hook.name][:, :, head_idx].mean(0)[None, ...]
    elif scheme == "zero":
        activations[:, :, head_idx] = 0.
    return activations

def ablate_mlp(activations, hook, scheme, new_cache=None):
    assert scheme in ["mean", "zero"], "the ablation scheme should be either 'mean' or 'zero'"

    if scheme == "mean":
        assert new_cache is not None, "`new_cache` is required when mean ablating"
        activations[:, :, :] = new_cache[hook.name][:, :, :].mean(0)[None, ...]
    elif scheme == "zero":
        activations[:, :, :] = 0.
    return activations


def auto_circuit(model, threshold, val_logits, val_tokens, patching_cache, ablation_scheme="mean", include_mlps=False):
    """
    Identifies the nodes of `model` relevant to the task elicited by `val_tokens` 
    according to a given `threshold`.

    Parameters
    ----------
    - `model`: GPT2 TransformerLens model
    - `threshold`: If patching a node does not raise the KL divergence by
    at least this value, it will be ditched.
    - `val_logits`: Baseline logits used to compute the KL divergence on the first
    iteration. It has shape `(batch_size, seq_len, d_vocab)`
    - `val_tokens`: Tensor of shape `(batch_size, seq_len)` used to compute the 
    KL divergence at each iteration
    - `patching_cache`: `ActivationCache` containing the cached activations of 
    our model on the patching tokens.
    - `ablation_scheme`: either "mean" or "zero" for mean or zero ablation
    - `include_mlps`: whether to ignore the MLPs (`False`) or try to identify
    the relevant MLPs (`True`) 

    Returns
    -------
    - `circuit_heads`: List of included attention heads with the format
    [[layer, head], ...]
    - `circuit_mlps`: List of included MLPs in the format [layer, ...]
    """
    model.reset_hooks(including_permanent=True)
    circuit_heads = []
    circuit_mlps = []

    baseline_logprobs = F.log_softmax(val_logits.clone(), dim=-1).clone()
    score_func = partial(kl_div, baseline_logprobs=baseline_logprobs)

    curr_logits = val_logits.clone()

    # Traverse the nodes from downstream to upstream
    for layer in reversed(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # temporarily remove node
            hook_fn = partial(
                ablate_head, head_idx=head, scheme=ablation_scheme, new_cache=patching_cache
                )
            model.add_hook(utils.get_act_name("result", layer), hook_fn, is_permanent=False)
            temp_logits = model(val_tokens).clone()
            
            if (score_func(temp_logits).mean(0) - score_func(curr_logits).mean(0)) < threshold:
                # if the KL divergence does not increase over a threshold, the node
                # is not important, so remove permanently
                model.add_hook(utils.get_act_name("result", layer), hook_fn, is_permanent=True)
                curr_logits = temp_logits.clone()
            else:
                # include node in the circuit
                circuit_heads.append([layer, head])
            model.reset_hooks(including_permanent=False)
        
        if include_mlps:
            # repeat for MLP
            # temporarily remove node
            hook_fn = partial(
                ablate_mlp, scheme=ablation_scheme, new_cache=patching_cache
                )
            model.add_hook(utils.get_act_name("mlp_out", layer), hook_fn, is_permanent=False)
            temp_logits = model(val_tokens).clone()

            if (score_func(temp_logits).mean(0) - score_func(curr_logits).mean(0)) < threshold:
                # if the KL divergence does not increase over a threshold, the node
                # is not important, so remove permanently
                model.add_hook(utils.get_act_name("mlp_out", layer), hook_fn, is_permanent=True)
                curr_logits = temp_logits.clone()
            else:
                # include node in the circuit
                circuit_mlps.append(layer)
            model.reset_hooks(including_permanent=False)
        else:
            circuit_mlps = [layer for layer in range(model.cfg.n_layers)]
            
    return circuit_heads, circuit_mlps