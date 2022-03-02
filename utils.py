from __future__ import absolute_import, division, print_function

import random

import numpy as np
import torch
import sparsemax_loss
import relu_loss
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.stats import entropy
from itertools import zip_longest
from pytorch_transformers import PreTrainedTokenizer
from run_lm_finetuning import TextDataset
from entmax import SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss, sparsemax, entmax15, entmax_bisect


def compute_jsd(p, q, base=np.e):
    p, q = np.asarray(p.cpu()), np.asarray(q.cpu())
    p, q = p / p.sum(), q / q.sum()
    m = 1. / 2 * (p + q)
    ent = entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.
    if ent == float('Inf'):
        ent = torch.log(torch.FloatTensor([2]))
    return ent


def compute_sp(p, target):
    p = np.asarray(p.cpu())
    target = target.cpu()
    return 1 - (0.5 * np.linalg.norm(p)**2 - p[target] + 0.5)


def softmax_temperature(X, temperature=1.0, axis=None):
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


def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def load_and_cache_examples(cfg, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = cfg.eval_data_file if evaluate else cfg.train_data_file
    dataset = TextDataset(tokenizer, cfg, file_path=file_path, block_size=cfg.block_size)
    return dataset


def set_seed(cfg, n_gpu):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)


def relu_criterion(z=-1, y=-1, ignore_index=0, vocab_size=50257) -> float:
    if ignore_index == -1:
        return relu_criterion

    mse = torch.nn.MSELoss(reduction='sum')
    y_hat = F.relu(z)
    diff = mse(z, F.one_hot(y, num_classes=vocab_size).float().cuda()) - mse(z, y_hat)
    return 0.5 * diff / z.shape[0]


def sparsemax_criterion(z=-1, y=-1, ignore_index=0, vocab_size=50257):
    if ignore_index == -1:
        return relu_criterion

    mse = torch.nn.MSELoss(reduction='sum')
    y_hat = sparsemax(z)
    diff = mse(z, F.one_hot(y, num_classes=vocab_size).float().cuda()) - mse(z, y_hat)
    return 0.5 * diff / z.shape[0]


def mask_tokens(inputs, tokenizer: PreTrainedTokenizer, cfg):
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


def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p):
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None
    for i in range(continuation_length):
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :]
        prev = logits.argmax(dim=1, keepdim=True)
        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits


def ul_seq(model, batch, cfg):
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


def ngram_repeat_mask(xs, n):
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


def get_criterion_and_gen_func(cfg):
    loss_funcs = {
        "cross_entropy": nn.CrossEntropyLoss(ignore_index=-1),
        "sparsemax": SparsemaxLoss(k=cfg.entmax_k, ignore_index=-1),
        "entmax15": Entmax15Loss(k=cfg.entmax_k, ignore_index=-1),
        "entmax": EntmaxBisectLoss(alpha=cfg.entmax_alpha, n_iter=cfg.entmax_bisect_iter, ignore_index=-1),
        "entmax_alpha": "entmax_alpha",
        "relu": relu_loss.ReluLoss(ignore_index=-1),
        "my_sparsemax": sparsemax_loss.SparsemaxLoss(k=cfg.entmax_k, ignore_index=-1)
    }

    gen_funcs = {
        "cross_entropy": torch.softmax,
        "sparsemax": partial(sparsemax, k=cfg.entmax_k),
        "entmax15": partial(entmax15, k=cfg.entmax_k),
        "entmax": partial(
            entmax_bisect,
            alpha=cfg.entmax_alpha,
            n_iter=cfg.entmax_bisect_iter),
        "entmax_alpha": "entmax_alpha",
        "relu": F.relu,
        "my_sparsemax": partial(sparsemax_loss.sparsemax, k=cfg.entmax_k)
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
