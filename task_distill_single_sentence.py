#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.nn.functional as F

from easytokenizer import AutoTokenizer
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import BertModel
from transformer.optimization import BertAdam

from general_distill import working_dir
from general_distill import PregeneratedDataset as TrainingDataset
from task_distill_sentence_pairs import EvalDataset, EvalCollate


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Task Distillation")
    parser.add_argument("--training_file", type=str, required=True)
    parser.add_argument("--metrics_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        help="The full path of evaluation data file.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization.")
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage.")
    parser.add_argument("--from_scratch",
                        action='store_true',
                        help="Whether to train from scratch.")
    parser.add_argument("--epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Linear warmup proportion over total steps.")
    parser.add_argument("--logging_steps",
                        default=100,
                        type=int,
                        help="The interval steps to logging.")
    parser.add_argument("--save_steps",
                        default=500,
                        type=int,
                        help="The interval steps to save checkpoints.")
    parser.add_argument("--save_best",
                        action="store_true",
                        help="Whether to save checkpoint on best evaluation performance.")
    parser.add_argument("--cosine_reduction",
                        default=0,
                        type=float,
                        help="The reduction of simbert cosine similarity.")
    parser.add_argument("--magnification",
                        default=100,
                        type=int,
                        help="Magnification of cosine similarity mse loss.")
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # set device
    n_gpu = torch.cuda.device_count()
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    # set seed
    set_seed(args.seed)
    
    # load tokenizer and model
    tokenizer = AutoTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    teacher_model = BertModel.from_pretrained(args.teacher_model)
    if args.from_scratch:
        student_model = BertModel.from_scratch(args.student_model)
    else:
        student_model = BertModel.from_pretrained(args.student_model)
    
    student_model.to(device)
    teacher_model.to(device)
    if n_gpu > 1:
        if args.local_rank == -1:
            student_model = nn.DataParallel(student_model)
            teacher_model = nn.DataParallel(teacher_model)
        else:
            student_model = DistributedDataParallel(
                student_model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)
            teacher_model = DistributedDataParallel(
                teacher_model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)
    
    # build dataloader
    train_dataset = TrainingDataset(args.training_file,
                                    args.metrics_file,
                                    tokenizer,
                                    seq_len=args.max_seq_length,
                                    reduce_memory=args.reduce_memory)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler)
    # STS-B evaluation data file
    if args.eval_file:
        eval_dataset = EvalDataset(args.eval_file, tokenizer,
                                   max_seq_len=args.max_seq_length)
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=args.eval_batch_size,
                                     shuffle=False,
                                     collate_fn=EvalCollate(pad_token_id=tokenizer.pad_id()))
    # before training
    num_training_steps = len(train_dataloader) * args.epochs
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         schedule="warmup_linear",
                         warmup=args.warmup_proportion,
                         t_total=num_training_steps,
                         max_grad_norm=args.max_grad_norm)
    
    global_step = 0
    best_metrics = 0
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    
    student_model.train()
    teacher_model.eval()
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader, start=1):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with torch.no_grad():
                _, t_vecs = teacher_model(input_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)

            s_vecs = student_model(input_ids, attention_mask=attention_mask,
                                   output_all_encoded_layers=False)
            
            batch_size = s_vecs.shape[0]
            t_vecs = F.normalize(t_vecs, p=2, dim=-1)
            s_vecs = F.normalize(s_vecs, p=2, dim=-1)
            t_matrix = torch.matmul(t_vecs, t_vecs.transpose(0, 1))
            s_matrix = torch.matmul(s_vecs, s_vecs.transpose(0, 1))
            del t_vecs, s_vecs

            loss = torch.sum((s_matrix - t_matrix) * (s_matrix - t_matrix))
            loss = args.magnification * loss / (batch_size * batch_size)
            del t_matrix, s_matrix
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                time_diff = time.time() - tic_train
                logger.info("global step: %d, epoch: %d, batch: %d, loss: %.4f, time cost: %.2fs" %
                            (global_step, epoch, step, loss, time_diff))
            
            if (global_step % args.save_steps == 0 or global_step == num_training_steps) and args.local_rank in [-1, 0]:
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                if args.eval_file:
                    spearman_corr, pearson_corr = evaluate(student_model, eval_dataloader, device)
                    logging.info("spearman corr: %.4f, pearson corr: %.4f" % (spearman_corr, pearson_corr))
                    if args.save_best:
                        if best_metrics < spearman_corr:
                            best_metrics = spearman_corr
                            logging.info("Saving student model")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                        continue
                logging.info("Saving student model")
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
            

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model performance on a given dataset.
    Compute spearman correlation coefficient.
    """
    model.eval()
    similarities, labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        a_input_ids, a_attention_mask, b_input_ids, b_attention_mask, label = batch
        _, a_vecs = model(a_input_ids, attention_mask=a_attention_mask,
                          output_all_encoded_layers=False)
        _, b_vecs = model(b_input_ids, attention_mask=b_attention_mask,
                          output_all_encoded_layers=False)
        a_vecs = F.normalize(a_vecs, p=2, dim=-1)
        b_vecs = F.normalize(b_vecs, p=2, dim=-1)
        similarity = torch.sum(a_vecs * b_vecs, dim=-1)
        similarities.extend(similarity.tolist())
        labels.extend(label.tolist())
    spearman_corr = stats.spearmanr(similarities, labels)[0]
    pearson_corr = stats.pearsonr(similarities, labels)[0]
    model.train()
    return (spearman_corr, pearson_corr)


if __name__ == "__main__":
    args = parse_args()
    train(args)
    shutil.rmtree(working_dir, ignore_errors=True)
