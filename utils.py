from collections import defaultdict
import time

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

from easy_transformer.ioi_dataset import IOIDataset

TOKEN_TO_YEAR = {405: 0, 486: 1, 2999: 2, 3070: 3, 3023: 4, 2713: 5, 3312: 6, 2998: 7, 2919: 8, 2931: 9, 940: 10, 1157: 11, 1065: 12, 1485: 13, 1415: 14, 1314: 15, 1433: 16, 1558: 17, 1507: 18, 1129: 19, 1238: 20, 2481: 21, 1828: 22, 1954: 23, 1731: 24, 1495: 25, 2075: 26, 1983: 27, 2078: 28, 1959: 29, 1270: 30, 3132: 31, 2624: 32, 2091: 33, 2682: 34, 2327: 35, 2623: 36, 2718: 37, 2548: 38, 2670: 39, 1821: 40, 3901: 41, 3682: 42, 3559: 43, 2598: 44, 2231: 45, 3510: 46, 2857: 47, 2780: 48, 2920: 49, 1120: 50, 4349: 51, 4309: 52, 4310: 53, 4051: 54, 2816: 55, 3980: 56, 3553: 57, 3365: 58, 3270: 59, 1899: 60, 5333: 61, 5237: 62, 5066: 63, 2414: 64, 2996: 65, 2791: 66, 3134: 67, 3104: 68, 3388: 69, 2154: 70, 4869: 71, 4761: 72, 4790: 73, 4524: 74, 2425: 75, 4304: 76, 3324: 77, 3695: 78, 3720: 79, 1795: 80, 6659: 81, 6469: 82, 5999: 83, 5705: 84, 5332: 85, 4521: 86, 5774: 87, 3459: 88, 4531: 89, 3829: 90, 6420: 91, 5892: 92, 6052: 93, 5824: 94, 3865: 95, 4846: 96, 5607: 97, 4089: 98, 2079: 99}
TOKEN_TO_YEAR = defaultdict(int, TOKEN_TO_YEAR)

################################
#   SCORE/METRIC FUNCTIONS     #
################################

def compute_accuracy(model, val_tokens, val_answer_tokens, task="acronyms"):
    if task == "acronyms":
        return (model(val_tokens)["logits"][:, -1].argmax(-1) == val_answer_tokens[:, -1]).float().mean().item()
    if task == "ioi":
        return (model(val_tokens)["logits"][:, -1].argmax(-1) == val_answer_tokens).float().mean().item()
    if task == "greater-than":
        preds = [TOKEN_TO_YEAR[t.item()] for t in model(val_tokens)["logits"][:, -1].argmax(-1)]
        val_answer_str_tokens = [TOKEN_TO_YEAR[t.item()] for t in val_answer_tokens]
        return sum([int(pred) > int(gt) for pred, gt in zip(preds, val_answer_str_tokens)]) / val_answer_tokens.shape[0]

def kl_div(logits, baseline_logprobs):
    # logits and baseline_logprobs have shape [batch_size, seq_len, d_vocab]
    logits = logits[:, -1]
    baseline_logprobs = baseline_logprobs[:, -1]
    return F.kl_div(F.log_softmax(logits, dim=-1), baseline_logprobs, log_target=True, reduction="none").sum(dim=-1).mean()

def compute_logit_diff_acronym(logits, answer_tokens, average=True, letter=3):
    """
    Compute the logit difference between the correct answer and the largest logit
    of all the possible incorrect capital letters. This is done for every iteration
    (i.e. each of the three letters of the acronym) 
    Parameters:
    -----------
    - `logits`: `Tensor[batch_size, seq_len, d_vocab]`
    - `answer_tokens`: Tensor[batch_size, 3]
    - `letter`: Can be either 1, 2 or 3, depending on the letter of the acronym that
                we are trying to predict
    """
    # Logits of the correct answers (batch_size, 3)
    correct_logits = logits[:, -3:].gather(-1, answer_tokens[..., None]).squeeze()
    # Retrieve the maximum logit of the possible incorrect answers
    capital_letters_tokens = torch.tensor([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
         50, 51, 52, 53, 54, 55, 56, 57], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    batch_size = logits.shape[0]
    capital_letters_tokens_expanded = capital_letters_tokens.expand(batch_size, 3, -1)
    incorrect_capital_letters = capital_letters_tokens_expanded[capital_letters_tokens_expanded != answer_tokens[..., None]].reshape(batch_size, 3, -1)
    incorrect_logits, _ = logits[:, -3:].gather(-1, incorrect_capital_letters).max(-1)
    # Return the mean logit difference at the token position corresponding to the `letter`th letter
    logit_diff = (correct_logits - incorrect_logits)[..., -4 + letter].mean()
    return logit_diff.mean() if average else logit_diff


##########################
# LOADING DATA FUNCTIONS #
##########################

# the class and function to obtain the greater-than data is borrowed from the ACDC code 
# (https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/greaterthan/utils.py#L61)

NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

class GreaterThanConstants:
    YEARS: list[str]
    YEARS_BY_CENTURY: dict[str, list[str]]
    TOKENS: list[int]
    INV_TOKENS: dict[int, int]
    TOKENS_TENSOR: torch.Tensor
    INV_TOKENS_TENSOR: torch.Tensor

    _instance = None

    @classmethod
    def get(cls: type["GreaterThanConstants"], device) -> "GreaterThanConstants":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, model):
        _TOKENIZER = model.tokenizer

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [_TOKENIZER.encode(f" {str(year)[:2]}")[0], _TOKENIZER.encode(str(year)[2:])[0]]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {
            i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(0, 100)
        }
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(0, 100)], dtype=torch.long)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR


def get_year_data(num_examples, model):
    constants = GreaterThanConstants.get(model)

    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    #torch.random.manual_seed(54)
    nouns_perm = torch.randint(0, len(NOUNS), (num_examples,))
    years_perm = torch.randint(0, len(constants.YEARS), (num_examples,))

    prompts = []
    prompts_tokenized = []
    for i in range(num_examples):
        year = constants.YEARS[years_perm[i]]
        prompts.append(
            template.format(
                noun=NOUNS[nouns_perm[i]],
                year1=year,
            ) + year[:2]
        )
        prompts_tokenized.append(model.tokenizer.encode(prompts[-1], return_tensors="pt").to(model.cfg.device))
        assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
    prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
    assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

    return prompts_tokenized, prompts


def load_gpt2_tl():
    model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    return model


def get_data(n_patching=100, n_val=100, task="acronyms"):
    """
    Prepares the dataset and model that will be used to perform the experiments.

    Parameters
    ----------
    - `task`: ["acronyms", "ioi", "greater-than"]

    Returns
    -------
    - `dict` with keys `model`, `
    """

    assert task in ["acronyms", "ioi", "greater-than"], "task must be either of 'acronyms', 'ioi', 'greater-than'"

    data = {}

    model = load_gpt2_tl()

    if task == "acronyms":
        with open("data/acronyms_2_common.txt", "r") as f:
            prompts, acronyms = list(zip(*[line.split(", ") for line in set(f.read().splitlines())]))
        # giga-cursed way of sampling from the dataset
        patching_prompts, patching_acronyms = prompts[:n_patching], acronyms[:n_patching]
        val_prompts, val_acronyms = prompts[n_patching:n_patching+n_val], acronyms[n_patching:n_patching+n_val]
        patching_tokens = model.to_tokens(patching_prompts)
        patching_answer_tokens = model.to_tokens(patching_acronyms, prepend_bos=False)
        val_tokens = model.to_tokens(val_prompts)
        val_answer_tokens = model.to_tokens(val_acronyms, prepend_bos=False)

        gt_circuit = [[8, 11], [10, 10], [9, 9], [11, 4], [4, 11], [1, 0], [2, 2], [5, 8]]

    if task == "ioi":
        N = n_patching+n_val
        ioi_dataset = IOIDataset(
            prompt_type="BABA",
            N=N,
            tokenizer=model.tokenizer,
            nb_templates=1,
            prepend_bos=False,
        ) 
        patching_tokens = ioi_dataset.toks[:n_patching, :-1]
        patching_answer_tokens = ioi_dataset.toks[:n_patching, -1]

        val_tokens = ioi_dataset.toks[n_patching:n_patching+n_val, :-1]
        val_answer_tokens = ioi_dataset.toks[n_patching:n_patching+n_val, -1]

        # heads of the IOI circuit (see paper)
        gt_circuit = [
            [9, 9], [9, 6], [10, 0], # name mover heads
            [10, 7], [11, 10],       # negative name mover heads 
            [9, 0], [9, 7], [10, 1], [10, 2], [10, 6], [10, 10], [11, 2], [11, 9], # backup name mover heads
            [7, 3], [7, 9], [8, 6], [8, 10], # S-inhibition heads
            [5, 5], [6, 9], # induction heads
            [2, 2], [4, 11], # previous token heads
            [0, 1], [3, 0] # duplicate token heads
        ]

    if task == "greater-than":
        tokens, _ = get_year_data(n_patching+n_val, model)
        patching_tokens = tokens[:n_patching]
        patching_answer_tokens = tokens[:n_patching, -3]

        val_tokens = tokens[n_patching:n_patching+n_val]
        val_answer_tokens = tokens[n_patching:n_patching+n_val, -3]

        gt_circuit = [[5, 1], [5, 5], [6, 1], [6, 9], [7, 10], [8, 8], [8, 11], [9, 1]] # we're only including attention heads, omitting MLPs for now

    patching_logits, patching_cache = model.run_with_cache(patching_tokens)
    val_logits, val_cache = model.run_with_cache(val_tokens)

    data["patching_tokens"] = patching_tokens
    data["patching_answer_tokens"] = patching_answer_tokens
    data["patching_logits"] = patching_logits.detach()
    data["patching_cache"] = patching_cache
    data["val_tokens"] = val_tokens
    data["val_answer_tokens"] = val_answer_tokens
    data["val_logits"] = val_logits.detach()
    data["val_cache"] = val_cache

    data["gt_circuit"] = gt_circuit

    del model

    return data


def measure_time(model, val_tokens):
    n_repeats = 5

    ts = []
    for _ in range(n_repeats):
        ti = time.time()
        model(val_tokens)
        dt = time.time() - ti
        ts.append(dt)
    t = sum(ts)/5
    return t