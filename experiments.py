from itertools import product
from tqdm import tqdm
import pandas as pd

import torch

from transformers import AutoModelForCausalLM

from discovery_utils import auto_circuit
from pruning_utils import prune_model
from utils import load_gpt2_tl, compute_accuracy, get_data, measure_time


def hyperparameter_experiments(task):    
    thresholds = 10**torch.linspace(-5, 0, 30)

    n_patching = 250
    n_val = 250

    data = get_data(n_patching=n_patching, n_val=n_val, task=task)

    patching_tokens = data["patching_tokens"].cuda()
    patching_cache = data["patching_cache"]

    val_tokens = data["val_tokens"].cuda()
    val_answer_tokens = data["val_answer_tokens"].cuda()
    val_logits = data["val_logits"]

    data = []

    for ablation_scheme, include_mlps in tqdm(product(["mean", "zero"], [False, True]), total=4):
        for threshold in thresholds:
            acc, n_params, n_heads, n_mlp = prune_experiment(threshold, val_logits, 
                                            val_tokens, val_answer_tokens, 
                                            patching_tokens, patching_cache,
                                            task=task,
                                            ablation_scheme=ablation_scheme,
                                            include_mlps=include_mlps)
            data.append([ablation_scheme, include_mlps, threshold.item(), acc, n_params, n_heads, n_mlp])
            print(f"threshold: {threshold.item():.2e}, acc: {acc:.2f}, n_params: {n_params:.2e}, n_heads: {n_heads}, n_mlp: {n_mlp}")

    df = pd.DataFrame(data, columns=["ablation_scheme", "include_mlps", "threshold", "accuracy", "size", "n_heads", "n_mlp"])
    return df


def prune_experiment(threshold, val_logits, val_tokens, val_answer_tokens, 
                     patching_tokens, patching_cache, task, ablation_scheme="mean", 
                     include_mlps=False):
    embedding_parameters = (50257 * 768) + (1024 * 768)
    model = load_gpt2_tl()
    circuit_heads, circuit_mlps = auto_circuit(
        model, threshold, val_logits, val_tokens, patching_cache, 
        ablation_scheme=ablation_scheme, include_mlps=include_mlps)
    del model
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", 
                                                 output_hidden_states=False, 
                                                 use_cache=False).cuda()
    model.eval()
    model = prune_model(model, circuit_heads, circuit_mlps, 
                        patching_tokens, ablation_scheme=ablation_scheme)
    acc = compute_accuracy(model, val_tokens, val_answer_tokens, task=task)
    n_params = model.num_parameters() - embedding_parameters
    return acc, n_params, len(circuit_heads), len(circuit_mlps)


def benchmark_experiment(threshold, val_logits, val_tokens, val_answer_tokens, 
                     patching_tokens, patching_cache, task, ablation_scheme="mean", 
                     include_mlps=False):
    embedding_parameters = (50257 * 768) + (1024 * 768)
    model = load_gpt2_tl()
    circuit_heads, circuit_mlps = auto_circuit(
        model, threshold, val_logits, val_tokens, patching_cache, 
        ablation_scheme=ablation_scheme, include_mlps=include_mlps)
    del model
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", 
                                                 output_hidden_states=False, 
                                                 use_cache=False).cuda()
    model.eval()
    baseline_acc = compute_accuracy(model, val_tokens, val_answer_tokens, task=task)
    baseline_size = model.num_parameters() - embedding_parameters
    baseline_time = measure_time(model, val_tokens)
    model = prune_model(model, circuit_heads, circuit_mlps, 
                        patching_tokens, ablation_scheme=ablation_scheme)
    acc = compute_accuracy(model, val_tokens, val_answer_tokens, task=task)
    n_params = model.num_parameters() - embedding_parameters
    time = measure_time(model, val_tokens)
    return acc, (baseline_acc - acc), n_params, (baseline_size - n_params) / baseline_size, time, (baseline_time - time) / baseline_time