# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
import csv
from tqdm import tqdm


class Dataset_(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',  # 前缀，即论文里的special token
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            'index' : index,
            'qid': example['qid'],
            'question_text': example['question'],
            'question' : question,  # 加上前缀'question:'的文本
            'target' : None,  # 加上后缀' </s>'
            'passages' : passages,  # psg前拼上title，并分别加上前缀'title: '和'context: '
            'scores' : scores
        }


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator_(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        qid = [ex['qid'] for ex in batch]
        q_text = [ex['question_text'] for ex in batch]
        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]  # context前拼上question，都已加上前缀
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, None, None, passage_ids, passage_masks, qid, q_text)


def load_data_(wiki_path, data_path=None, global_rank=-1, world_size=-1, mrqa=False):
    def load_passages(ctx_file: str):
        # all_passages = load_passages("/zhengmq2010/odqa_data/open_domain_data/psgs_w100.tsv")
        # print(f'wiki_psgs loading finished!!')
        docs = {}
        print('Reading data from: %s', ctx_file)
        with open(ctx_file, encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in tqdm(reader):
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
        return docs

    def get_psg_text(data):
        all_passages = load_passages(wiki_path)
        for i, d in enumerate(data):
            for ctx in d['ctxs']:
                ctx['title'] = all_passages[ctx['id']][1]
                ctx['text'] = all_passages[ctx['id']][0]
        return data

    with open(data_path, 'r') as fin:
        data = json.load(fin)
    if isinstance(data, list):  # 官方文件（uGAR和pyserini？），只需修改wiki id
        if 'wiki' in str(data[0]['ctxs'][0]['id']):
            for i, d in enumerate(data):
                d['qid'] = str(i)
                for ctx in d['ctxs']:
                    ctx['id'] = ctx['id'][5:]
        else:
            for i, d in enumerate(data):
                d['qid'] = str(i)
        if 'text' not in data[0]['ctxs'][0].keys():
            data = get_psg_text(data)
        return data

    examples = []
    for k, example in enumerate(data.items()):
        example[1]['qid'] = example[0]
        for c in example[1]['ctxs']:
            c['score'] = c['original_score']
        examples.append(example[1])
    if 'text' not in data['1']['ctxs'][0].keys():
        examples = get_psg_text(examples)

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
