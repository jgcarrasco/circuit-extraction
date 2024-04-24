from functools import partial
from pathlib import Path
from tqdm import tqdm
import time

from string import ascii_uppercase
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import plotly.express as px

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaSdpaAttention
from transformers.cache_utils import Cache


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100
SIZE = 2000

def sort_Nd_tensor(tensor, descending=False):
    i = torch.sort(tensor.flatten(), descending=descending).indices
    return np.array(np.unravel_index(i.numpy(), tensor.shape)).T.tolist()


def compute_logit_diff(logits, answer_tokens, 
                       capital_letters_tokens, average=True):
    """
    Compute the logit difference between the correct answer and the largest logit
    of all the possible incorrect capital letters. This is done for every iteration
    (i.e. each of the three letters of the acronym) and then averaged if desired.
    If `average=False`, then a `Tensor[batch_size, 3]` is returned, containing the
    logit difference at every iteration for every prompt in the batch

    Parameters:
    -----------
    - `logits`: `Tensor[batch_size, seq_len, d_vocab]`
    - `answer_tokens`: Tensor[batch_size, 3]
    """
    # Logits of the correct answers (batch_size, 3)
    correct_logits = logits[:, -3:].gather(-1, answer_tokens[..., None]).squeeze()
    # Retrieve the maximum logit of the possible incorrect answers
    batch_size = logits.shape[0]
    capital_letters_tokens_expanded = capital_letters_tokens.expand(batch_size, 3, -1)
    incorrect_capital_letters = capital_letters_tokens_expanded[capital_letters_tokens_expanded != answer_tokens[..., None]].reshape(batch_size, 3, -1)
    incorrect_logits, _ = logits[:, -3:].gather(-1, incorrect_capital_letters).max(-1)
    # Return the mean
    return (correct_logits - incorrect_logits).mean() if average else (correct_logits - incorrect_logits)


def compute_accuracy(logits, answer_tokens, 
                     capital_letters_tokens):
    """
    Computes the accuracy among the possible outputs (specified by `capital_letters_tokens`).
    Specifically, it retrieves the predicted token among the vocabulary and checks if it is
    the same as `answer_tokens[-1]`.

    Parameters:
    -----------
    - `logits`: `Tensor[batch_size, seq_len, d_vocab]`
    - `answer_tokens`: Tensor[batch_size, 3]
    """
    # Retrieve logits of the letters of the vocabulary
    logits_vocab = logits[:, -1, capital_letters_tokens]
    # Now the indices [0, 1, 2, 3,...] represent [capital_letter_tokens[0], ...]
    # Transform the answer tokens to [0, 1, 2, ...]
    answer_token_t = torch.tensor([torch.where(capital_letters_tokens == element)[0][0] for element in answer_tokens[:, -1]]).to(device)
    # Get predictions (max logit)
    preds = logits_vocab.argmax(-1)
    acc = (preds == answer_token_t).float().mean()
    std = (preds == answer_token_t).float().std()
    return acc, std


class AcronymDataset(Dataset):
    def __init__(self, path: str, tokenizer, size=None):
        with open(path, "r") as f:
            prompts, acronyms = list(zip(*[line.split(", ") for line in f.read().splitlines()]))
        self.prompts = prompts
        self.acronyms = acronyms
        self.cap_to_id = {k: v[0] for k, v in zip(ascii_uppercase, tokenizer(list(ascii_uppercase), add_special_tokens=False)["input_ids"])}

        self.tokens = tokenizer(prompts, return_tensors="pt")["input_ids"]
        self.answer_tokens = torch.tensor([[self.cap_to_id[c] for c in acronym] for acronym in acronyms])
    
        if size:
            self.prompts = self.prompts[:size]
            self.acronyms = self.acronyms[:size]
            self.tokens = self.tokens[:size]
            self.answer_tokens = self.answer_tokens[:size]

    def __len__(self):
        return self.tokens.shape[0]
    
    def __getitem__(self, idx):
        return self.tokens[idx], self.answer_tokens[idx]
    

def get_cache_attn(model, tokens) -> torch.Tensor:
    def get_cache_attn_hook(module: LlamaSdpaAttention, input, output, layer_idx: int, cache: torch.Tensor):
        """
        Caches the mean activation of the attention layer. `cache` has shape (n_layer, d_model)
        output -> tensor of shape (batch_size, seq_len, d_model)
        """
        cache[layer_idx] = output[0].mean(0).detach().clone()

    cache_attn = torch.zeros((model.config.num_hidden_layers, 11, model.config.hidden_size), dtype=torch.float16).to(device)

    for layer in range(model.config.num_hidden_layers):  
        hook_fn = partial(get_cache_attn_hook, layer_idx=layer, cache=cache_attn)
        hook = model.model.layers[layer].self_attn.register_forward_hook(hook_fn)
        corrupted_logits = model(tokens)["logits"]
        hook.remove()
    
    return cache_attn

def get_cache_attn_full(model, dataloader):
    cache_attn = []
    for tokens, _ in dataloader:
        cache_attn.append(get_cache_attn(model_hf, tokens.cuda()).cpu())
    return torch.stack(cache_attn, dim=0).mean(0)


def get_attribution_score_attn(model, val_tokens, val_answer_tokens, cache) -> torch.Tensor:
    def mean_ablate_attn_hf(module: LlamaSdpaAttention, input, output, layer_idx: int, cache: torch.Tensor):
        """
        Performs mean ablation on an attention layer
        This function 

        output -> tuple containing (output[0], None)
        output[0] -> tensor of shape (batch_size, seq_len, d_model)
        cache -> tensor of shape (n_layers, seq_len, d_model)
        """
        return (cache[layer_idx][None, ...], None, output[2])


    corrupted_logit_diffs = torch.zeros((model.config.num_hidden_layers, val_tokens.shape[0]))
    with torch.no_grad():
        for layer in range(model.config.num_hidden_layers):
            hook_fn = partial(mean_ablate_attn_hf, layer_idx=layer, cache=cache)
            hook = model.model.layers[layer].self_attn.register_forward_hook(hook_fn)
            corrupted_logits = model(val_tokens)["logits"]
            hook.remove()
            corrupted_logit_diff = compute_logit_diff(corrupted_logits.cuda(), val_answer_tokens, capital_letters_tokens, average=False)
            corrupted_logit_diffs[layer] = corrupted_logit_diff[..., -1] # take last letter

    attribution_score_attn_hf = (corrupted_logit_diffs - val_logit_diff_hf.cpu()).mean(-1)
    return attribution_score_attn_hf

def get_attribution_score_attn_full(model, dataloader, cache):
    attribution_score_attn_hf = []
    for val_tokens, val_answer_tokens in dataloader:
        attribution_score_attn_hf.append(get_attribution_score_attn(model, val_tokens.cuda(), val_answer_tokens.cuda(), cache.cuda()).cpu())
    return torch.stack(attribution_score_attn_hf, dim=0).mean(0)


def get_cache_mlp(model, tokens) -> torch.Tensor:

    def get_cache_mlp_hook(module: LlamaMLP, input, output, layer_idx: int, cache: torch.Tensor):
        """
        Caches the mean activation of the MLP. `cache` has shape (n_layer, d_model)
        output -> tensor of shape (batch_size, seq_len, d_model)
        """
        cache[layer_idx] = output.mean(0).detach().clone()

    cache_mlp = torch.zeros((model.config.num_hidden_layers, 11, model.config.hidden_size), dtype=torch.float16).to(device)

    for layer in range(model.config.num_hidden_layers):  
        hook_fn = partial(get_cache_mlp_hook, layer_idx=layer, cache=cache_mlp)
        hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
        corrupted_logits = model(tokens)["logits"]
        hook.remove()
    return cache_mlp


def get_attribution_score_mlp(model, val_tokens, val_answer_tokens, cache) -> torch.Tensor:

    def mean_ablate_mlp_hf(module: LlamaMLP, input, output, layer_idx: int, cache: torch.Tensor):
        """
        Performs mean ablation on an MLP layer

        output -> tensor of shape (batch_size, seq_len, d_model)
        cache -> tensor of shape (n_layers, seq_len, d_model)
        """
        output = cache[layer_idx][None, ...]
        return output

        
    corrupted_logit_diffs = torch.zeros((model.config.num_hidden_layers, val_tokens.shape[0]))
    with torch.no_grad():
        for layer in range(model.config.num_hidden_layers):
            hook_fn = partial(mean_ablate_mlp_hf, layer_idx=layer, cache=cache)
            hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
            corrupted_logits = model(val_tokens)["logits"]
            hook.remove()
            corrupted_logit_diff = compute_logit_diff(corrupted_logits.cuda(), val_answer_tokens, capital_letters_tokens, average=False)
            corrupted_logit_diffs[layer] = corrupted_logit_diff[..., -1] # take last letter

    attribution_score_mlp_hf = (corrupted_logit_diffs - val_logit_diff_hf.cpu()).mean(-1)
    return attribution_score_mlp_hf

def get_attribution_score_mlp_full(model, dataloader, cache):
    attribution_score_mlp_hf = []
    for val_tokens, val_answer_tokens in dataloader:
        attribution_score_mlp_hf.append(get_attribution_score_mlp(model, val_tokens.cuda(), val_answer_tokens.cuda(), cache.cuda()).cpu())
    return torch.stack(attribution_score_mlp_hf, dim=0).mean(0)


def get_cache_mlp_full(model, dataloader):
    cache_mlp = []
    for tokens, _ in dataloader:
        cache_mlp.append(get_cache_mlp(model_hf, tokens.cuda()).cpu())
    return torch.stack(cache_mlp, dim=0).mean(0)


class PrunedRMSNorm(nn.Module):
    """
    Replaces the RMSNorm with an identity function.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states):
        return hidden_states


class PrunedMLP(nn.Module):
    """
    Layer that simply returns the vector `mean_activation`, which is expected
    to have shape (seq_len, d_model) and represents the mean activation of the MLP
    that is going to be replaced.
    """
    def __init__(self, mean_activation: torch.Tensor):
        super().__init__()
        self.mean_activation = mean_activation

    def forward(self, x):
        return self.mean_activation[None, ...]
        

class PrunedAttention(nn.Module):
    """
    Layer that simply returns the vector `mean_activation`, which is expected
    to have shape (seq_len, d_model) and represents the mean activation of the MLP
    that is going to be replaced.
    """
    def __init__(self, mean_activation: torch.Tensor):
        super().__init__()
        self.mean_activation = mean_activation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return (self.mean_activation[None, ...], None, past_key_value)
    

def find_component_to_ablate(model, val_dataloader, cache_attn, cache_mlp, ablated_heads):
    """
    1. Caches the mean outputs of each attn layer and MLP
    2. Uses the mean outputs to perform activation patching and checking which components 
       contribute positively/negatively to performance.
    3. Returns the component that contributes less to the performance [comp, layer], where comp=1 if MLP, 0 if attn
    """
    
    attribution_score_attn_hf = get_attribution_score_attn_full(model, val_dataloader, cache_attn)
    attribution_score_mlp_hf = get_attribution_score_mlp_full(model, val_dataloader, cache_mlp)
    # Row 0 contains scores for attn, row 1 for MLPs
    attribution_scores = torch.stack([attribution_score_attn_hf, attribution_score_mlp_hf], dim=0)
    # Set the attribution scores of the already pruned components to a super low value so that they never get selected
    if ablated_heads:
        for comp, layer in ablated_heads:
            attribution_scores[comp, layer] = -1e6
    sorted_components = sort_Nd_tensor(attribution_scores)
    # Get non-ablated component with the highest score 
    component_to_ablate = sorted_components[-1]
    return component_to_ablate


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent

    tokenizer_hf = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, use_cache=False).cuda()
    initial_parameters = model_hf.num_parameters()

    capital_letters_tokens = torch.tensor(tokenizer_hf(list(ascii_uppercase), add_special_tokens=False)["input_ids"], dtype=torch.long, device=device).squeeze()

    dataset = AcronymDataset(path=root_dir / "cache_acronyms.txt", tokenizer=tokenizer_hf, size=SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = AcronymDataset(path=root_dir / "val_acronyms.txt", tokenizer=tokenizer_hf, size=SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cap_to_id = {k: v[0] for k, v in zip(ascii_uppercase, tokenizer_hf(list(ascii_uppercase), add_special_tokens=False)["input_ids"])}

    tokens_hf, answer_tokens = next(iter(dataloader))
    val_tokens_hf, val_answer_tokens = next(iter(val_dataloader))

    logits_hf = model_hf(tokens_hf.cuda())["logits"]
    val_logits_hf = model_hf(val_tokens_hf.cuda())["logits"]

    logit_diff_hf = compute_logit_diff(logits_hf, answer_tokens.cuda(), capital_letters_tokens, average=False)[..., -1].mean()
    val_logit_diff_hf = compute_logit_diff(val_logits_hf, val_answer_tokens.cuda(), capital_letters_tokens, average=False)[..., -1].mean()

    acc_hf, std = compute_accuracy(logits_hf, answer_tokens.cuda(), capital_letters_tokens)
    val_acc_hf, val_std = compute_accuracy(val_logits_hf, val_answer_tokens.cuda(), capital_letters_tokens)

    print("BASELINES:")
    print("-"*40)
    print(f"Logit Diff: {logit_diff_hf.item():.4f}\t Val: {val_logit_diff_hf.item():.4f}")
    print(f"Accuracy: {acc_hf.item():.2f} ± {std:.2f}\t Val: {val_acc_hf.item():.2f} ± {val_std:.2f}")

    logit_diffs = []
    std_logit_diffs = []

    accs = []
    std_accs = []

    num_parameters = []

    ablated_components = []


    for i in tqdm(range(2 * model_hf.config.num_hidden_layers)):
        cache_attn = get_cache_attn_full(model_hf, dataloader)
        cache_mlp = get_cache_mlp_full(model_hf, dataloader)
        comp, layer = find_component_to_ablate(model_hf, val_dataloader, cache_attn, cache_mlp, ablated_components)
        print(f"Ablating {'MLP' if comp else 'Attn'} {layer}")
        ablated_components.append([comp, layer])
        if comp == 0:
            model_hf.model.layers[layer].input_layernorm = PrunedRMSNorm()
            model_hf.model.layers[layer].self_attn = PrunedAttention(mean_activation=cache_attn[layer].cuda())
        else:
            model_hf.model.layers[layer].post_attention_layernorm = PrunedRMSNorm()
            model_hf.model.layers[layer].mlp = PrunedMLP(mean_activation=cache_mlp[layer].cuda())
        num_parameters.append(model_hf.num_parameters())
        av_logit_diff = []
        std_logit_diff = []
        for val_tokens_hf, val_answer_tokens in val_dataloader:
            val_tokens_hf = val_tokens_hf.cuda()
            circuit_logits = model_hf(val_tokens_hf)["logits"]
            # Compute logit diff
            logit_diff = compute_logit_diff(circuit_logits.cuda(), val_answer_tokens.cuda(), capital_letters_tokens, average=False)
            av_logit_diff.append(logit_diff[..., -1].mean(0))
            std_logit_diff.append(logit_diff[..., -1].std(0))
        av_logit_diff = torch.stack(av_logit_diff, dim=0).mean(0)
        std_logit_diff = torch.stack(std_logit_diff, dim=0).mean(0)
        logit_diffs.append(av_logit_diff)
        std_logit_diffs.append(std_logit_diff)
        # Compute accuracy
        acc, std = compute_accuracy(circuit_logits.cuda(), val_answer_tokens.cuda(), capital_letters_tokens)
        print(f"Logit diff: {av_logit_diff}\tAcc: {acc}")
        accs.append(acc)
        std_accs.append(std)
    logit_diffs_hf = torch.stack(logit_diffs, dim=0)
    std_logit_diffs_hf = torch.stack(std_logit_diffs, dim=0)
    accs_hf = torch.stack(accs, dim=0)
    std_accs_hf = torch.stack(std_accs, dim=0)

    cutoff = (logit_diffs_hf >= logit_diff_hf).nonzero()[-1].item() # last point where the performance is equal or higher than baseline
    cutoff_acc = (accs_hf >= acc_hf).nonzero()[-1].item() # last point where the performance is equal or higher than baseline

    labels = [f"{'MLP' if comp else 'Attn.'} {layer}" for comp, layer in ablated_components]
    fig = px.line(x = labels, y=logit_diffs_hf.cpu().numpy(), 
                error_y=std_logit_diffs_hf.cpu().numpy(),
                labels={"y": "Logit Diff.", "x": "Component"}, title="Logit Diff. vs. Ablated Component")
    fig.add_hline(y=logit_diff_hf.item(), line_width=1.5, line_dash="dash", line_color="black")
    fig.add_vline(x=labels[cutoff], line_width=1., line_dash="dash", line_color="red")
    fig.write_image(root_dir / "images" / "logit_diff.pdf")
    time.sleep(1)
    fig.write_image(root_dir / "images" / "logit_diff.pdf")

    labels = [f"{'MLP' if comp else 'Attn.'} {layer}" for comp, layer in ablated_components]
    fig = px.line(x = labels, y=accs_hf.cpu().numpy(), 
                error_y=std_accs_hf.cpu().numpy(),
                labels={"y": "Accuracy", "x": "Component"}, title="Accuracy vs. Ablated Component")
    fig.add_hline(y=acc_hf.item(), line_width=1.5, line_dash="dash", line_color="black")
    fig.add_vline(x=labels[cutoff_acc], line_width=1., line_dash="dash", line_color="red")
    fig.write_image(root_dir / "images" / "accuracy.pdf")
    time.sleep(1)
    fig.write_image(root_dir / "images" / "accuracy.pdf")
   

    p_parameters = [x/initial_parameters for x in num_parameters]

    fig = px.line(x = p_parameters, y=logit_diffs_hf.cpu().numpy(), 
                error_y=std_logit_diffs_hf.cpu().numpy(),
                labels={"y": "Logit Diff.", "x": "% parameters"}, title="Logit Diff. vs. model size")
    fig.add_vline(x=p_parameters[cutoff], line_width=1., line_dash="dash", line_color="red")
    fig.add_hline(y=logit_diff_hf.item(), line_width=1.5, line_dash="dash", line_color="black")
    fig.update_xaxes(autorange="reversed")
    fig.write_image(root_dir / "images" / "logit_diff_size.pdf")
    time.sleep(1)
    fig.write_image(root_dir / "images" / "logit_diff_size.pdf")

    fig = px.line(x = p_parameters, y=accs_hf.cpu().numpy(), 
                error_y=std_accs_hf.cpu().numpy(),
                labels={"y": "Accuracy", "x": "% parameters"}, title="Accuracy vs. model size")
    fig.add_vline(x=p_parameters[cutoff_acc], line_width=1., line_dash="dash", line_color="red")
    fig.add_hline(y=acc_hf.item(), line_width=1.5, line_dash="dash", line_color="black")
    fig.update_xaxes(autorange="reversed")
    fig.write_image(root_dir / "images" / "accuracy_size.pdf")
    time.sleep(1)
    fig.write_image(root_dir / "images" / "accuracy_size.pdf")