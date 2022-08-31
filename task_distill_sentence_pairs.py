#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import numpy as np
import os
import random
import shutil
import tempfile
import time
import torch

from easytokenizer import AutoTokenizer
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import DistillBertForSemantics
from transformer.optimization import BertAdam


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

working_dir = tempfile.mkdtemp()


def convert_example_to_features(sentence, tokenizer, seq_len):
    input_ids, _ = tokenizer.encode(sentence, max_length=seq_len)
    attention_mask = [1] * len(input_ids)
    pad_len = seq_len - len(input_ids)
    if pad_len > 0:
        input_ids += [ tokenizer.pad_id() ] * pad_len
        attention_mask += [ 0 ] * pad_len
    return input_ids, attention_mask


class TrainingDataset(Dataset):
    def __init__(self, training_file, metrics_file, tokenizer, seq_len=128, reduce_memory=False):
        assert os.path.isfile(training_file)
        assert os.path.isfile(metrics_file)
        metrics_handle = open(metrics_file, mode="r", encoding="utf-8")
        metrics = json.load(metrics_handle)
        num_samples = metrics["num_samples"]
        #seq_len = metrics["max_seq_len"]
        
        if reduce_memory:
            a_input_ids = np.memmap(os.path.join(working_dir, "a_input_ids.memmap"),
                                    mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
            b_input_ids = np.memmap(os.path.join(working_dir, "b_input_ids.memmap"),
                                    mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
            a_attention_mask = np.memmap(os.path.join(working_dir, "a_attention_mask.memmap"),
                                         mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
            b_attention_mask = np.memmap(os.path.join(working_dir, "b_attention_mask.memmap"),
                                         mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
        else:
            a_input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            b_input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            a_attention_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            b_attention_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        
        with open(training_file, mode="r", encoding="utf-8") as data_handle:
            for i, line in enumerate(tqdm(data_handle, total=num_samples, desc="Training samples")):
                line = line.strip().split("\t")
                a_features = convert_example_to_features(line[0], tokenizer, seq_len)
                b_features = convert_example_to_features(line[1], tokenizer, seq_len)
                a_input_ids[i] = a_features[0]
                b_input_ids[i] = b_features[0]
                a_attention_mask[i] = a_features[1]
                b_attention_mask[i] = b_features[1]

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.a_input_ids = a_input_ids
        self.b_input_ids = b_input_ids
        self.a_attention_mask = a_attention_mask
        self.b_attention_mask = b_attention_mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return (torch.as_tensor(self.a_input_ids[index].astype(np.int64)),
                torch.as_tensor(self.a_attention_mask[index].astype(np.int64)),
                torch.as_tensor(self.b_input_ids[index].astype(np.int64)),
                torch.as_tensor(self.b_attention_mask[index].astype(np.int64)))
        

class EvalDataset(Dataset):
    def __init__(self, eval_file, tokenizer, max_seq_len=128):
        a_input_ids, b_input_ids, labels = [], [], []
        with open(eval_file, mode="r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip().split("\t")
                if len(line) != 3:
                    continue
                label = float(line[2]) if '.' in line[2] else int(line[2])
                a_features = tokenizer.encode(line[0], max_length=max_seq_len)
                b_features = tokenizer.encode(line[1], max_length=max_seq_len)
                a_input_ids.append(a_features[0])
                b_input_ids.append(b_features[0])
                labels.append(label)
        
        self.labels = labels
        self.a_input_ids = a_input_ids
        self.b_input_ids = b_input_ids
        self.a_attention_mask = [[1] * len(_) for _ in a_input_ids]
        self.b_attention_mask = [[1] * len(_) for _ in b_input_ids]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (self.a_input_ids[index], self.a_attention_mask[index],
                self.b_input_ids[index], self.b_attention_mask[index],
                self.labels[index])
        

class EvalCollate:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        a_input_ids = [item[0] for item in batch]
        a_attention_mask = [item[1] for item in batch]
        b_input_ids = [item[2] for item in batch]
        b_attention_mask = [item[3] for item in batch]
        labels = [item[4] for item in batch]
        
        # padding
        a_max_seq_len = max(len(l) for l in a_input_ids)
        for i in range(len(a_input_ids)):
            pad_len = a_max_seq_len - len(a_input_ids[i])
            a_input_ids[i] += [self.pad_token_id] * pad_len
            a_attention_mask[i] += [0] * pad_len
        
        b_max_seq_len = max(len(l) for l in b_input_ids)
        for i in range(len(b_input_ids)):
            pad_len = b_max_seq_len - len(b_input_ids[i])
            b_input_ids[i] += [self.pad_token_id] * pad_len
            b_attention_mask[i] += [0] * pad_len
        
        return (torch.as_tensor(a_input_ids), torch.as_tensor(a_attention_mask),
                torch.as_tensor(b_input_ids), torch.as_tensor(b_attention_mask),
                torch.as_tensor(labels))


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
    teacher_model = DistillBertForSemantics.from_pretrained(args.teacher_model)
    if args.from_scratch:
        student_model = DistillBertForSemantics.from_scratch(args.student_model)
    else:
        student_model = DistillBertForSemantics.from_pretrained(args.student_model)
    
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
    
    loss_fct = nn.MSELoss()
    student_model.train()
    teacher_model.eval()
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader, start=1):
            batch = tuple(t.to(device) for t in batch)
            a_input_ids, a_attention_mask, b_input_ids, b_attention_mask = batch
            with torch.no_grad():
                t_out = teacher_model(a_input_ids, b_input_ids,
                                      a_attention_mask=a_attention_mask,
                                      b_attention_mask=b_attention_mask)
                t_out -= args.cosine_reduction
            
            s_out = student_model(a_input_ids, b_input_ids,
                                  a_attention_mask=a_attention_mask,
                                  b_attention_mask=b_attention_mask)

            loss = loss_fct(t_out.view(-1), s_out.view(-1))
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
        similarity = model(a_input_ids, b_input_ids,
                           a_attention_mask=a_attention_mask,
                           b_attention_mask=b_attention_mask)
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
