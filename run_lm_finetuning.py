# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import re
import pickle
import hydra
import utils
from omegaconf import OmegaConf


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, PreTrainedTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            cfg: OmegaConf,
            file_path: str,
            block_size=512
            ):

        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            cfg.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not cfg.overwrite_cache:
            logging.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                if hasattr(tokenizer, 'build_inputs_with_special_tokens'):
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
                else:
                    self.examples.append(tokenized_text[i: i + block_size])
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logging.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def train(
        cfg: OmegaConf,
        train_dataset: TextDataset,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        loss_func,
        gen_func,
        n_gpu=0,
        device=None) -> tuple:

    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(filename_suffix=utils.get_filename_suffix(cfg))

    train_batch_size = cfg.per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset) if cfg.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=False, sampler=train_sampler, batch_size=train_batch_size)

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=t_total)

    if cfg.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if cfg.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank],
                                                          output_device=cfg.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", cfg.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size * cfg.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if cfg.local_rank != -1 else 1))
    logging.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    def calculate_loss(logits, labels, loss_func):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = loss_func(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        return loss

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    model.train()
    model.zero_grad()
    utils.set_seed(cfg, n_gpu)  # Added here for reproducibility (even between python 2 and 3)s
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    best_jsd, best_ppl, best_sp = 100000, 100000, 0

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = utils.mask_tokens(batch, tokenizer, cfg) if cfg.mlm else (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            if cfg.unlikelihood_seq and torch.rand(1).item() < 0.5:
                if inputs.size(1) < 50:
                    continue
                else:
                    loss = utils.ul_seq(model, inputs, cfg)
            else:
                _, logits, _ = model(inputs) if cfg.mlm else model(inputs, labels=labels)
                loss = calculate_loss(logits, labels, loss_func)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            epoch_iterator.set_description(f"batch loss: {format(loss.item(), '.2f')}")
            tb_writer.add_scalar('train_loss', loss, global_step)

            if cfg.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if cfg.max_steps > 0 and global_step > cfg.max_steps:
                epoch_iterator.close()
                break

        # EVAL
        if cfg.local_rank in [-1, 0]:
            # Log metrics
            # Only evaluate when single GPU otherwise metrics may not average well
            if cfg.local_rank == -1 and cfg.evaluate_during_training:
                jsd, ppl, sp = evaluate(cfg, model, tokenizer, gen_func=gen_func, top_p=cfg.top_p)
                tb_writer.add_scalar('eval_jsd', jsd, epoch)
                tb_writer.add_scalar('eval_ppl', ppl, epoch)
                tb_writer.add_scalar('eval_sp', sp, epoch)

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss) / cfg.logging_steps, epoch)
            logging_loss = tr_loss

            # Save model checkpoint
            if jsd < best_jsd:
                best_jsd = jsd
                output_dir = os.path.join(cfg.output_dir+'/best_jsd', 'checkpoint')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(cfg, os.path.join(output_dir, 'training_args.bin'))
                logging.info("Saving model checkpoint to %s", output_dir)
            if ppl < best_ppl:
                best_ppl = ppl
                output_dir = os.path.join(cfg.output_dir+'/best_ppl', 'checkpoint')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(cfg, os.path.join(output_dir, 'training_args.bin'))
                logging.info("Saving model checkpoint to %s", output_dir)
            if sp > best_sp:
                best_sp = sp
                output_dir = os.path.join(cfg.output_dir+'/best_sp', 'checkpoint')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(cfg, os.path.join(output_dir, 'training_args.bin'))
                logging.info("Saving model checkpoint to %s", output_dir)

        if cfg.max_steps > 0 and global_step > cfg.max_steps:
            train_iterator.close()
            break

    if cfg.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
        cfg: OmegaConf,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        prefix="",
        gen_func=torch.softmax,
        n_gpu=0,
        device=None) -> tuple:

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = utils.load_and_cache_examples(cfg, tokenizer, evaluate=True)

    if not os.path.exists(cfg.output_dir) and cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.output_dir)

    eval_batch_size = cfg.per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if cfg.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", eval_batch_size)

    perp = 0.0
    model.eval()
    jsd, sp = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(device)

        with torch.no_grad():
            _, logits, _ = model(batch, masked_lm_labels=batch) if cfg.mlm else model(batch, labels=batch)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = batch[..., 1:].contiguous().squeeze(0)

            if cfg.temp != 0:
                probs = utils.softmax_temperature(shift_logits, temperature=cfg.temp, axis=1)
            elif cfg.top_p > 0 or cfg.top_k > 0:
                shift_logits = utils.top_k_top_p_filtering(
                    shift_logits,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    gen_func=gen_func)
                probs = gen_func(shift_logits, dim=1)
            else:
                probs = gen_func(shift_logits, dim=1)
            lprobs = probs

            if len(probs[0].nonzero()) != len(probs[0]):
                probs = probs[:, :]+cfg.epsilon
                sums = [probs[i].sum().item() for i in range(probs.size(0))]
                probs = [probs[i] / sums[i] for i in range(len(sums))]
                probs = torch.stack(probs)

            p = [probs[i, shift_labels.squeeze(0)[i]] for i in range(len(shift_labels.squeeze(0)))]
            p = torch.stack(p)
            perp += torch.log(p**(-1)).mean().item()

            jsd_batch = []
            labels = torch.zeros(len(shift_labels), shift_logits.size(-1))
            for i in range(len(shift_labels)):
                labels[i, shift_labels[i]] = 1
                jsd_ = utils.compute_jsd(lprobs[i], labels[i])
                if jsd_ != float('Inf'):
                    jsd_batch.append(jsd_)

            jsd_batch = torch.tensor(jsd_batch).mean()
            jsd += jsd_batch

            sp_batch = []
            for i in range(len(shift_labels)):
                sp_batch.append(utils.compute_sp(lprobs.squeeze(0)[i], shift_labels[i]))

            sp_batch = torch.tensor(sp_batch).mean()
            sp += sp_batch

    a = perp / len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(a))
    jsd = jsd / len(eval_dataloader)
    sp = sp / len(eval_dataloader)

    output_eval_file = os.path.join(cfg.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results {} *****".format(prefix))
        logging.info(f'perplexity: {perplexity}')
        logging.info(f'js: {jsd}')
        logging.info(f'sp; {sp}')
        writer.write(f'perplexity: {perplexity}\n')
        writer.write(f'js: {jsd}\n')
        writer.write(f'sp; {sp}\n')

    return jsd, perplexity, sp


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):

    if cfg.model_type in ["bert", "roberta"] and not cfg.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. "
                         "They must be run using the --mlm flag (masked language modeling).")
    if cfg.eval_data_file is None and cfg.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file "
                         "to --eval_data_file or remove the --do_eval argument.")

    if os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir) and cfg.do_train and not cfg.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(cfg.output_dir))

    # Setup distant debugging if needed
    if cfg.server_ip and cfg.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(cfg.server_ip, cfg.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    n_gpu, device = 0, None
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, 'train.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if cfg.local_rank in [-1, 0] else logging.WARN,
        encoding='utf-8',
    )
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    cfg.local_rank, device, n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    utils.set_seed(cfg, n_gpu)

    # Load pretrained model and tokenizer
    if cfg.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    # Set up model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.model_type]
    config = (
        config_class.from_pretrained(
            cfg.config_name if cfg.config_name else cfg.model_name_or_path) if cfg.mode == 'finetune'
        else config_class())
    tokenizer = tokenizer_class.from_pretrained(
        cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
        do_lower_case=cfg.do_lower_case)

    # Our input block size will be the max possible for the model
    cfg.block_size = tokenizer.max_len if cfg.block_size <= 0 else cfg.block_size
    loss_func, gen_func = utils.get_criterion_and_gen_func(cfg)

    if cfg.mode == 'finetune':
        model = model_class.from_pretrained(
            cfg.model_name_or_path,
            from_tf='.ckpt' in cfg.model_name_or_path,
            config=config)
    else:
        logging.info("Training new model from scratch")
        model = model_class(config=config)
    model.to(device)

    if cfg.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logging.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))

    # Training
    if cfg.do_train:
        if cfg.local_rank not in [-1, 0]:
            # Barrier to make sure only the first process in distributed training process the dataset,
            # and the others will use the cache
            torch.distributed.barrier()

        train_dataset = utils.load_and_cache_examples(cfg, tokenizer, evaluate=False)

        if cfg.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(
            cfg,
            train_dataset,
            model,
            tokenizer,
            loss_func,
            gen_func,
            n_gpu=n_gpu,
            device=device)

        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if cfg.do_train and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(cfg.output_dir) and cfg.local_rank in [-1, 0]:
            os.makedirs(cfg.output_dir)

        logging.info("Saving model checkpoint to %s", cfg.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(cfg, os.path.join(cfg.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(cfg.output_dir)
        tokenizer = tokenizer_class.from_pretrained(cfg.output_dir, do_lower_case=cfg.do_lower_case)
        model.to(device)

    # Evaluation
    if cfg.do_eval and cfg.local_rank in [-1, 0]:
        checkpoints = [cfg.output_dir]

        if cfg.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(cfg.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logging.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, loss=loss_func, gen_func=gen_func)
            model.to(device)

            evaluate(
                cfg,
                model,
                tokenizer,
                prefix=global_step,
                gen_func=gen_func,
                top_p=cfg.top_p,
                n_gpu=n_gpu,
                device=device)


if __name__ == "__main__":
    main()
