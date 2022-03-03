from __future__ import absolute_import, division, print_function

import random

import numpy as np
from omegaconf import OmegaConf
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.stats import entropy
from itertools import zip_longest
from pytorch_transformers import PreTrainedTokenizer
from entmax import SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss, sparsemax, entmax15, entmax_bisect


def calculate_loss(logits, labels, loss_func):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1))
    return loss


def compute_jsd(p, q, base=np.e) -> torch.tensor:
    p, q = p.cpu(), q.cpu()
    p = p.detach().numpy()
    q = q.detach().numpy()
    p, q = p / p.sum(), q / q.sum()
    m = 1. / 2 * (p + q)
    ent = entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.
    if ent == float('Inf'):
        ent = torch.log(torch.FloatTensor([2]))
    return ent


def compute_sp(p, target) -> float:
    p = np.asarray(p.cpu().detach().numpy())
    target = target.cpu()
    return 1 - (0.5 * np.linalg.norm(p)**2 - p[target] + 0.5)


def softmax_temperature(X, temperature=1.0, axis=None) -> torch.tensor:
    X = X.squeeze(0)
    for i in range(len(X)):
        X[i] = X[i] * (1 / temperature)
    p = torch.softmax(X, dim=-1)
    return p


def process_chunk(d):
    """Replace this with your own function
    that processes data one line at a
    time"""

    d = d.strip() + ' processed'
    return d


def grouper(n, iterable, padvalue=None) -> zip_longest:
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def set_seed(cfg, n_gpu):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)


def mask_tokens(inputs, tokenizer: PreTrainedTokenizer, cfg) -> tuple:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability cfg.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, cfg.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), gen_func=torch.softmax):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        cfg:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    if top_k > 0:
        for i in range(logits.size(0)):
            indices_to_remove = logits[i] < torch.topk(logits[i], top_k)[0][..., -1, None]
            logits[i][indices_to_remove] = filter_value

    for i in range(logits.size(0)):
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(gen_func(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = filter_value
    return logits


def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p) -> tuple:
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None
    for _ in range(continuation_length):
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :]
        prev = logits.argmax(dim=1, keepdim=True)
        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits


def ul_seq(model, batch, cfg) -> torch.tensor:
    input_sequence = batch.cuda()
    batch = model.batch_input_sequence_by_prefix_length(input_sequence, 50)
    completions, continuation_logits = sample_sequence(model, batch, 50, 100, cfg.top_k, cfg.top_p)
    pred_toks = completions[:, 50:].contiguous()
    mask = ngram_repeat_mask(pred_toks, 4).type_as(continuation_logits)
    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    loss = loss / ntokens
    return loss


def ngram_repeat_mask(xs, n) -> torch.tensor:
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()

        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)

    return mask


def get_criterion_and_gen_func(cfg) -> tuple:
    loss_funcs = {
        "cross_entropy": nn.CrossEntropyLoss(ignore_index=-1),
        "sparsemax": SparsemaxLoss(k=cfg.entmax_k, ignore_index=-1),
        "entmax15": Entmax15Loss(k=cfg.entmax_k, ignore_index=-1),
        "entmax": EntmaxBisectLoss(alpha=cfg.entmax_alpha, n_iter=cfg.entmax_bisect_iter, ignore_index=-1),
        "entmax_alpha": "entmax_alpha"
    }

    gen_funcs = {
        "cross_entropy": torch.softmax,
        "sparsemax": partial(sparsemax, k=cfg.entmax_k),
        "entmax15": partial(entmax15, k=cfg.entmax_k),
        "entmax": partial(
            entmax_bisect,
            alpha=cfg.entmax_alpha,
            n_iter=cfg.entmax_bisect_iter),
        "entmax_alpha": "entmax_alpha"
    }

    return loss_funcs[cfg.loss],  gen_funcs[cfg.loss]


def get_filename_suffix(cfg):
    if cfg.loss == 'entmax':
        return '_entmax_' + str(cfg.entmax_alpha)
    elif cfg.loss == 'entmax15':
        return '_entmax_1.5'
    elif cfg.loss == 'sparsemax':
        return '_sparsemax'
    elif cfg.loss == 'cross_entropy':
        return '_softmax'
    elif cfg.loss == 'entmax_alpha':
        return '_entmax_alpha'
    elif cfg.top_p > 0:
        return '_nucleus_' + str(cfg.top_p)
    elif cfg.loss == 'relu':
        return 'relu'
    elif cfg.loss == 'my_sparsemax':
        return 'my_sparsemax'


def repeat_at_1(predictions, targets, context_length: int, topk=0, topp=0.0) -> tuple:
    predictions = torch.tensor(predictions).cpu()
    targets = targets.unsqueeze(0)
    T = targets.size(1)
    assert predictions.size(0) == T

    # T x T where prev_targets[t, :] = [y_1,...,y_t-1, -1, -1,..., -1]
    prev_targets = targets.expand(T, T).tril().masked_fill_(
        torch.ones_like(targets.expand(T, T)).byte().triu().bool(),
        -1)

    # each row t is [-1, ..., -1, y_{t-k-1}, ..., y_{t-1}, -1, ..., -1] where k is context length
    prev_targets = prev_targets.masked_fill_(
        torch.ones_like(targets.expand(T, T)).byte().tril(-(context_length+1)).bool(),
        -1)
    prev_targets = torch.tensor(prev_targets).cpu()

    repeat_at_1 = (predictions[:, None] == prev_targets)
    has_repeat_at_1 = repeat_at_1.sum(1).gt(0)
    total_repeat_at_1 = has_repeat_at_1.sum()

    is_incorrect = (predictions != targets.view(-1)).view(-1, 1)
    total_wrong_repeat_at_1 = ((repeat_at_1 * is_incorrect).sum(1).gt(0)).sum()

    return total_repeat_at_1.item() / float(targets.size(1)), total_wrong_repeat_at_1.item()/float(targets.size(1))


def calculate_metrics(cfg: OmegaConf, logits: list, batch: list, gen_func, repeat=None, wrong_repeat=None) -> tuple:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = batch[..., 1:].contiguous().squeeze(0).view(-1)

    if cfg.temp != 0:
        probs = softmax_temperature(shift_logits, temperature=cfg.temp, axis=1)
    else:
        if cfg.top_p > 0 or cfg.top_k > 0:
            shift_logits = top_k_top_p_filtering(
                shift_logits,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                gen_func=gen_func)
        probs = gen_func(shift_logits, dim=1)
    lprobs = probs

    if len(probs[0].nonzero()) != len(probs[0]):
        probs = probs[:, :] + cfg.epsilon
        sums = [probs[i].sum().item() for i in range(probs.size(0))]
        probs = [probs[i] / sums[i] for i in range(len(sums))]
        probs = torch.stack(probs)

    p = [probs[i, shift_labels[i]] for i in range(len(shift_labels))]
    p = torch.stack(p)

    sp_batch, jsd_batch = [], []
    labels = torch.zeros(len(shift_labels), shift_logits.size(-1))
    for i in range(len(shift_labels)):
        labels[i, shift_labels[i]] = 1
        jsd_ = compute_jsd(lprobs[i], labels[i])

        if not math.isinf(jsd_):
            jsd_batch.append(jsd_)

        sp_batch.append(compute_sp(lprobs.squeeze(0)[i], shift_labels[i]))

    perp = torch.log(p**(-1)).mean()
    jsd = torch.tensor(jsd_batch).mean()
    sp = torch.tensor(sp_batch).mean()

    if repeat is not None and wrong_repeat is not None:
        pred = torch.multinomial(lprobs, num_samples=1).squeeze(1).view(-1).tolist()
        for context_length in [16, 32, 128, 512]:
            cur_repeat, cur_wrong_repeat = repeat_at_1(
                pred, shift_labels, context_length,
                topk=cfg.top_k, topp=cfg.top_p)
            repeat[context_length].append(cur_repeat)

            wrong_repeat[context_length].append(cur_wrong_repeat)

    return jsd, perp, sp
