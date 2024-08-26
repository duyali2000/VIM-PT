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


from __future__ import absolute_import
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import pickle
import csv
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
import time
from model import Seq2Seq
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)
from data.dataset import get_dataset

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "Javascript":"js"}

def initParser():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )
    parser.add_argument("--source_lang", default="Java", type=str,
                        help="The language of input")
    parser.add_argument("--target_lang", default="Java", type=str,
                        help="The language of input")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_comment_length", default=32, type=int,
                        help="The maximum total comment sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature        ', type=float, default=0.1)
    parser.add_argument('--lingual_number', type=int, default=7)
    parser.add_argument('--weightAB', type=float, default=1)
    parser.add_argument('--weightBB', type=float, default=0)
    parser.add_argument('--weightcon', type=float, default=0)
    parser.add_argument('--autonum', type=int, default=0)
    parser.add_argument('--lamda', type=float, default=0.01)
    parser.add_argument('--special_length', type=int, default=16)
    return parser


    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = initParser()
    args = parser.parse_args()
    logger.info(args) # print arguments
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('cuda is not available')
        assert 0
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Set seed
    set_seed(args.seed)

    config = AutoConfig.from_pretrained("microsoft/codebert-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    encoder = AutoModel.from_pretrained("microsoft/codebert-base")

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id, args=args)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.n_gpu > 1: # multi-gpu training
        model = torch.nn.DataParallel(model)


    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = get_dataset(nam="train", tokenizer=tokenizer, special_length=args.special_length)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size, num_workers = 4)
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)

        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
             
        dev_dataset, src_best_bleu, tgt_best_bleu, best_loss = {}, 0, 0, 1e6
        # Initialize, load all the training samples into the memory bank for random sampling 
        if args.load_model_path is None:
            state = 'initialize'
            for i, batch in enumerate(train_dataloader, 0):
                #source_ids, source_masks, labels = tuple(t.to(device) for t in batch)
                idxs, source_ids, source_masks, labels = batch
                source_ids, source_masks, labels = source_ids.to(device), source_masks.to(device), labels.to(device)
                model(state, args, source_ids, source_masks, labels)

        # auto-encoding, pre-train target decoder
        for autoepoch in range(args.autonum):
            time0 = time.time()
            model.train()
            state, tr_loss = 'auto', 0
            for i, batch in enumerate(train_dataloader, 0):
                #source_ids, source_masks, labels = tuple(t.to(device) for t in batch)
                idxs, source_ids, source_masks, labels = batch
                source_ids, source_masks, labels = source_ids.to(device), source_masks.to(device), labels.to(device)
                loss = model(state, args, source_ids, source_masks, labels)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                time1 = time.time()
                print("Autoencoding epoch %d, batch loss: %.4f, time cost %.2fs" % (autoepoch, tr_loss / (i + 1), time1 - time0))
            time1 = time.time()
            print("Autoencoding epoch %d, epoch loss: %.4f, time cost %.2fs" % (autoepoch, tr_loss, time1 - time0))

        # joint traning and evaluation
        global_step = len(train_dataloader)*args.num_train_epochs
        for epoch in range(args.num_train_epochs):
            time0 = time.time()
            model.train()
            state, tr_loss = 'train', 0
            for i, batch in enumerate(train_dataloader, 0):
                #source_ids, source_masks, labels = tuple(t.to(device) for t in batch)
                idxs, source_ids, source_masks, labels = batch
                source_ids, source_masks, labels = source_ids.to(device), source_masks.to(device), labels.to(device)
                for src_lingual in range(args.lingual_number):
                    warm_up_step = (i + epoch * args.num_train_epochs) / global_step
                    loss = model(state, args, source_ids, source_masks, labels, src_lingual, warm_up_step)
                    if args.n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    tr_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                time1 = time.time()
                print("epoch %d, batch loss: %.4f, time cost %.2fs" % (epoch, tr_loss / (i+1), time1 - time0))
            time1 = time.time()
            print("epoch %d, epoch loss: %.4f, time cost %.2fs" % (epoch, tr_loss, time1-time0))

            if args.do_eval and (epoch+1) % 5 == 0:
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples, eval_data = get_dataset(nam="test", tokenizer=tokenizer, special_length=args.special_length)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                args.eval_batch_size = len(eval_examples)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)
                model.eval()

                dev_bleu_all = 0
                for batch in eval_dataloader:
                    idxs, source_ids, source_masks, labels = batch
                    source_ids, source_masks, labels = source_ids.to(device), source_masks.to(device), labels.to(device)
                    with torch.no_grad():
                        for src_lingual in range(args.lingual_number):
                            preds_dict = model('test', args, source_ids, source_masks, labels, src_lingual)
                            for tgt_lingual in range(args.lingual_number):
                                if(src_lingual == tgt_lingual):
                                    continue
                                preds = preds_dict[str(src_lingual) + '-' + str(tgt_lingual)]
                                tgt_p = []
                                for p in preds:
                                    t=p[0].cpu().numpy().tolist()
                                    if 0 in t:
                                        t=t[:t.index(0)]
                                    tgt_p.append(tokenizer.decode(t,clean_up_tokenization_spaces=False))
                                #tgt_dict[str(src_lingual) + '-' + str(tgt_lingual)] = tgt_p

                                tmp = list(postfix.keys())
                                source_lang = tmp[src_lingual]
                                target_lang = tmp[tgt_lingual]
                                generate_path = source_lang + '-' + target_lang + "-test.output"
                                with open(os.path.join(args.output_dir, generate_path), 'w') as f1:
                                    #tgts = tgt_dict[str(src_lingual) + '-' + str(tgt_lingual)]
                                    #for ref in tgts:
                                    for ref in tgt_p:
                                        f1.write(ref + '\n')
                                truth_path = './data/test_truth/test.' + postfix[target_lang]
                                with open(truth_path, 'w') as f2:
                                    for idx in idxs:
                                        instance = eval_examples[idx]
                                        source_code = instance[target_lang]
                                        f2.write(source_code + '\n')
                                dev_bleu = round(_bleu(truth_path,
                                                       os.path.join(args.output_dir, generate_path)), 2)
                                logger.info("%s to %s:  %s = %s " % (source_lang, target_lang, "bleu-4", str(dev_bleu)))
                                fcsv = open('result-lamda-'+str(args.lamda)+'.csv', 'a+', encoding='utf-8')
                                csv_writer = csv.writer(fcsv)
                                csv_writer.writerow(
                                    [epoch, source_lang, target_lang, args.lamda, dev_bleu])
                                fcsv.close()
                                dev_bleu_all += dev_bleu

                if dev_bleu_all > tgt_best_bleu:
                    tgt_best_bleu = dev_bleu_all
                    # save best checkpoint
                    best_output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                    if not os.path.exists(best_output_dir):
                        os.makedirs(best_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(best_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()

