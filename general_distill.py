#!usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import math
import numpy as np
import os
import random
import shutil
import tempfile
import time
import torch
import torch.nn.functional as F

from easytokenizer import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import BertModel
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


class PregeneratedDataset(Dataset):
    def __init__(self, training_file, metrics_file, tokenizer, seq_len=128, reduce_memory=False):
        assert os.path.isfile(training_file)
        assert os.path.isfile(metrics_file)
        metrics_handle = open(metrics_file, mode="r", encoding="utf-8")
        metrics = json.load(metrics_handle)
        num_samples = metrics["num_samples"]
        #seq_len = metrics["max_seq_len"]

        if reduce_memory:
            input_ids = np.memmap(os.path.join(working_dir, "input_ids.memmap"),
                                  mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
            attention_mask = np.memmap(os.path.join(working_dir, "attention_mask.memmap"),
                                       mode="w+", dtype=np.int32, shape=(num_samples, seq_len))
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            attention_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        
        with open(training_file, mode="r", encoding="utf-8") as data_handle:
            for i, line in enumerate(tqdm(data_handle, total=num_samples, desc="Training samples")):
                line = line.strip()
                features = convert_example_to_features(line, tokenizer, seq_len)
                input_ids[i] = features[0]
                attention_mask[i] = features[1]
                
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return (torch.as_tensor(self.input_ids[index].astype(np.int64)),
                torch.as_tensor(self.attention_mask[index].astype(np.int64)))


def parse_args():
    parser = argparse.ArgumentParser(description="General Distillation")
    parser.add_argument("--training_file", type=str, required=True)
    parser.add_argument("--metrics_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--teacher_layer_index",
                        default=-1,
                        type=int,
                        help="The transformer layer index of teacher model to distill.")
    parser.add_argument("--student_layer_index",
                        default=-1,
                        type=int,
                        help="The transformer layer index of student model to distill.")
    parser.add_argument("--num_relation_heads",
                        default=12,
                        type=int,
                        help="The number of relation heads for distillation.")
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
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Whether to train from scratch.")
    parser.add_argument("--epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage.")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
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
    args = parser.parse_args()
    return args


def calc_minilm_loss(loss_fct, s, t, attn_mask, num_relation_heads=0):
    # Initialize head_num
    if num_relation_heads > 0 and num_relation_heads != s.shape[1]:
        # new shape: [batch_size, seq_len, head_num, head_dim]
        s = s.permute(0, 2, 1, 3)

        head_dim_new = int(s.shape[2] * s.shape[3] / num_relation_heads)
        # new shape: [batch_size, seq_len, num_relation_heads, head_dim_new]
        new_shape = (s.shape[0], s.shape[1], num_relation_heads, head_dim_new)
        s = s.reshape(new_shape)
        
        # new shape: [batch_size, num_relation_heads, seq_len, head_dim_new]
        s = s.permute(0, 2, 1, 3)
    
    if num_relation_heads > 0 and num_relation_heads != t.shape[1]:
        t = t.permute(0, 2, 1, 3)
        head_dim_new = int(t.shape[2] * t.shape[3] / num_relation_heads)
        new_shape = (t.shape[0], t.shape[1], num_relation_heads, head_dim_new)
        t = t.reshape(new_shape)
        t = t.permute(0, 2, 1, 3)
    
    s_head_dim, t_head_dim = s.shape[3], t.shape[3]
    s_r = torch.matmul(s, s.transpose(-1, -2)) / math.sqrt(s_head_dim)
    s_r += attn_mask
    del s
    t_r = torch.matmul(t, t.transpose(-1, -2)) / math.sqrt(t_head_dim)
    t_r += attn_mask
    del t
    loss = loss_fct(F.log_softmax(s_r, dim=-1), F.softmax(t_r, dim=-1))
    return loss
    
    
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
    train_dataset = PregeneratedDataset(args.training_file, 
                                        args.metrics_file,
                                        tokenizer,
                                        seq_len=args.max_seq_length,
                                        reduce_memory=args.reduce_memory)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler)
    
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
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    loss_fct = nn.KLDivLoss(reduction="sum", log_target=False)
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
                _, _, q_t, k_t, v_t = teacher_model(input_ids, 
                    attention_mask=attention_mask, 
                    output_all_encoded_layers=False, output_qkv=True,
                    layer_index=args.teacher_layer_index)
            
            _, _, q_s, k_s, v_s = student_model(input_ids, 
                attention_mask=attention_mask,
                output_all_encoded_layers=False, output_qkv=True,
                layer_index=args.student_layer_index)
            
            batch_size = q_t.shape[0]
            pad_seq_len = q_t.shape[2]
            attention_mask = (1 - attention_mask) * -1e9
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)       
            # query-query relation
            qr_loss = calc_minilm_loss(loss_fct, q_s, q_t, attention_mask, args.num_relation_heads)
            del q_s, q_t
            # key-key relation
            kr_loss = calc_minilm_loss(loss_fct, k_s, k_t, attention_mask, args.num_relation_heads)
            del k_s, k_t
            # value-value relation
            vr_loss = calc_minilm_loss(loss_fct, v_s, v_t, attention_mask, args.num_relation_heads)
            del v_s, v_t
            
            loss = qr_loss + kr_loss + vr_loss
            loss /= (args.num_relation_heads * pad_seq_len * batch_size)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            if global_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                time_diff = time.time() - tic_train
                logger.info("global step: %d, epoch: %d, batch: %d, loss: %.4f, time cost: %.2fs" %
                            (global_step, epoch, step, loss, time_diff))
                #tic_train = time.time()
            
            if (global_step % args.save_steps == 0 or global_step == num_training_steps) and args.local_rank in [-1, 0]:
                logging.info("Saving student model")
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                #tic_train = time.time()                
            
    
if __name__ == "__main__":
    args = parse_args()
    train(args)
    shutil.rmtree(working_dir, ignore_errors=True)
