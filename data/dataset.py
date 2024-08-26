import csv
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
import pickle
from itertools import cycle
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)
from preprocess import summary_replace, code_replace

postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "Javascript":"js"}
example = {"Java":None, "C#":None, "C++":None, "C":None, "Python":None, "PHP":None, "Javascript":None}
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def load_dataset(name):
    if(name=="train"):
        with open("./data/train_label_data.pkl", mode="rb") as f:
            a = pickle.load(f)
        with open("./data/val_label_data.pkl", mode="rb") as f:
            b = pickle.load(f)
        with open("/home/xxx/multitranslation/publish/data/train_snippet_label_data.pkl", mode="rb") as f:
            c = pickle.load(f)
        with open("/home/xxx/multitranslation/publish/data/val_snippet_label_data.pkl", mode="rb") as f:
            d = pickle.load(f)
        return a|b|c|d
        #return Merge(a, b)
    else:
        with open("./data/"+name+"_label_data.pkl", mode="rb") as f:
            a = pickle.load(f)
        return a


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, instance_ids, instance_masks, labels):
        self.example_id = example_id
        self.instance_ids = instance_ids
        self.instance_masks = instance_masks
        self.labels = labels


def convert_examples_to_features(examples, tokenizer, special_length):
    features = []
    for idx in examples.keys():
        instance_ids = []
        instance_masks = []
        instance_labels = []
        instance = examples[idx]
        #comment = summary_replace(instance["comment"])
        comment = instance["comment"]
        comment_tokens = tokenizer.tokenize(comment)[: 64 - 1]
        comments = comment_tokens + [tokenizer.sep_token]
        padding_length = 64 - len(comments)
        comment_ids = tokenizer.convert_tokens_to_ids(comments) + [tokenizer.pad_token_id] * padding_length
        comment_mask = [1] * len(comments) + [0] * padding_length
        for lingual in postfix.keys():
            source_code = instance[lingual]
            if((lingual == "C") and (source_code == None)):
                source_code = instance["C++"]
            if(source_code == None):
                instance_labels.append(0)
                padding_length = 512 - 64
                source_code_ids = [tokenizer.pad_token_id] * padding_length
                source_code_mask = [0] * padding_length
            else:
                #source_code = code_replace(source_code)
                instance_labels.append(1)
                source_tokens = tokenizer.tokenize(source_code)[: (512 - 64 - 2 - special_length)]
                source_codes = [tokenizer.unk_token] * special_length + [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
                padding_length = 512 - 64 - len(source_codes)
                source_code_ids = tokenizer.convert_tokens_to_ids(source_codes) + [tokenizer.pad_token_id] * padding_length
                source_code_mask = [1] * len(source_codes) + [0] * padding_length

            source_ids = source_code_ids + comment_ids
            source_mask = source_code_mask + comment_mask

            instance_ids.append(source_ids)
            instance_masks.append(source_mask)

        features.append(InputFeatures(example_id=idx,
                                      instance_ids=instance_ids, instance_masks=instance_masks, labels=instance_labels))
    return features


class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ids_tensors, masks_tensors, labels_tensors = [], [], []
        for ids in self.examples[item].instance_ids:
            ids_tensors.append(torch.tensor(ids))
        for masks in self.examples[item].instance_masks:
            masks_tensors.append(torch.tensor(masks))
        for label in self.examples[item].labels:
            labels_tensors.append(torch.tensor(label))
        ids_tensors = torch.stack(ids_tensors)
        masks_tensors = torch.stack(masks_tensors)
        labels_tensors = torch.stack(labels_tensors)
        #print(ids_tensors.shape, masks_tensors.shape, labels_tensors.shape)
        return self.examples[item].example_id, ids_tensors, masks_tensors, labels_tensors


def get_dataset(nam, tokenizer, special_length):
    examples = load_dataset(name=nam)
    train_features = convert_examples_to_features(examples, tokenizer, special_length)
    data = TextDataset(train_features)
    return examples, data



