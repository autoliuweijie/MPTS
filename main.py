# coding: utf-8
import os
import sys
import argparse

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm
from sklearn import metrics
from typing import Callable, Iterable, Optional, Tuple, Union

from utils import (
    get_linear_schedule_with_warmup,
    AdamW
)


transformers.logging.set_verbosity_error()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MPTSDataset(Dataset):

    perspectives = [
            'Action', 'Adventure', 'Animation', 'Comedy',
            'Crime', 'Drama', 'Family', 'Fantasy',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller'
        ]


    def __init__(self,
                 path,
        ):
        super().__init__()
        self.path = path

        self.data = []
        with open(path, 'r') as fin:
            for i, line in enumerate(fin):
                if i == 0:
                    columns = line.strip().split('\t')
                    col2idx = {col: i for i, col in enumerate(columns)}
                    id_col = col2idx['id']
                    texta_col = col2idx['texta']
                    textb_col = col2idx['textb']
                else:
                    items = line.strip().split('\t')
                    sample = {
                        'id': items[id_col],
                        'texta': items[texta_col],
                        'textb': items[textb_col],
                    }
                    for k in self.perspectives:
                        v = int(items[col2idx[k]])
                        sample[k] = 1 if v == 1 else 0
                    self.data.append(sample)

        self.persp2idx = {p: i for i, p in enumerate(self.perspectives)}
        self.idx2persp = {i: p for i, p in enumerate(self.perspectives)}

    def __len__(self):
        return len(self.data)

    def stastics(self):
        support = {k: 0 for k in self.perspectives}
        for s in self.data:
            for k in self.perspectives:
                if s[k] == 1:
                    support[k] += 1
        return support

    def __getitem__(self,
                    index
        ):
        return self.data[index]

    @classmethod
    def collate_fn(cls,
                   batch):
        batch_collated = {
            'id': [s['id'] for s in batch],
            'texta': [s['texta'] for s in batch],
            'textb': [s['textb'] for s in batch],
        }
        for k in cls.perspectives:
            batch_collated[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
        return batch_collated

    @classmethod
    def collate_fn_with_label(cls,
                              batch):
        batch_collated = {
            'id': [s['id'] for s in batch],
            'texta': [s['texta'] for s in batch],
            'textb': [s['textb'] for s in batch],
            'label': torch.tensor([[s[k] for k in cls.perspectives] for s in batch], dtype=torch.float32)
        }

        return batch_collated


class Encoder(nn.Module):

    def __init__(self,
                 args
        ):
        super().__init__()
        self.args = args
        self.pool_type = args.pool_type
        self.persp_num = args.persp_num
        self.device = args.device

    def forward(self,
                texta,
                textb,
                labels=None
        ):
        raise NotImplementedError("Encoder can not be called.")

    def _pooling(self,
                 model_output,
                 attention_mask,
                 pool_type
        ):
        last_hidden    = model_output.last_hidden_state
        hidden_states  = model_output.hidden_states

        if pool_type == 'cls':
            pooled_result = last_hidden[:, 0]
        elif pool_type == 'avg':
            pooled_result = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif pool_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden   = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif pool_type == 'avg_first_last':
            first_hidden  = hidden_states[0]
            last_hidden   = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        return pooled_result


class CrossEncoder(Encoder):

    def __init__(self,
                 args
        ):
        super().__init__(args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.init_model_dir)
        self.model = AutoModel.from_pretrained(args.init_model_dir)
        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.persp_num)
        self.linear1.to(self.device)
        self.linear2.to(self.device)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss(reduction='mean')

    def forward(self,
                texta,
                textb,
                labels=None
        ):

        inputs = self.tokenizer(texta, textb, padding=True, truncation=True, max_length=self.args.max_length, return_tensors='pt')
        inputs = inputs.to(self.device)
        model_output = self.model(**inputs, output_hidden_states=True)
        embs = self._pooling(model_output, inputs.attention_mask, self.pool_type)  # batch_size x hidden_size
        embs = self.linear1(embs)
        embs = self.linear2(embs)  # batch_size x persp_num
        sim_scores = self.sigmoid(embs)  # batch_size x persp_num

        if labels is None:
            predict_label = torch.gt(sim_scores, 0.5).int()
            loss = None
            acc  = None
        else:
            predict_label = torch.gt(sim_scores, 0.5).int()
            sim_scores_flatten = sim_scores.flatten()
            labels_flatten = labels.flatten().to(self.device)
            loss = self.loss(sim_scores_flatten, labels_flatten)
            acc = torch.mean((predict_label.flatten() == labels_flatten).float())

        return sim_scores, predict_label, loss, acc

    def save_model(self,
                   model_dir
        ):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)

        linear1_path = os.path.join(model_dir, 'linear1.bin')
        linear2_path = os.path.join(model_dir, 'linear2.bin')
        linear1_state_dict = self.linear1.state_dict()
        linear2_state_dict = self.linear2.state_dict()
        torch.save(linear1_state_dict, linear1_path)
        torch.save(linear2_state_dict, linear2_path)
        return model_dir

    def load_model(self,
                   model_dir
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.to(self.device)

        linear1_path = os.path.join(model_dir, 'linear1.bin')
        linear2_path = os.path.join(model_dir, 'linear2.bin')
        linear1_state_dict = torch.load(linear1_path, map_location=self.device)
        linear2_state_dict = torch.load(linear2_path, map_location=self.device)
        self.linear1.load_state_dict(linear1_state_dict, strict=True)
        self.linear2.load_state_dict(linear2_state_dict, strict=True)
        return model_dir


class BiEncoder(Encoder):

    def __init__(self,
                 args
        ):
        super().__init__(args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.init_model_dir)
        self.model = AutoModel.from_pretrained(args.init_model_dir)
        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size
        self.temperature = self.args.temperature

        self.multi_persp_linears = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.persp_num)])
        self.multi_persp_linears.to(args.device)

        self.loss = nn.BCELoss(reduction='mean')

    def forward(self,
                texta,
                textb,
                labels=None
        ):
        inputs_a = self.tokenizer(texta, padding=True, truncation=True, max_length=self.args.max_length, return_tensors='pt')
        inputs_b = self.tokenizer(textb, padding=True, truncation=True, max_length=self.args.max_length, return_tensors='pt')
        inputs_a = inputs_a.to(self.device)
        inputs_b = inputs_b.to(self.device)

        model_outputs_a = self.model(**inputs_a, output_hidden_states=True)
        model_outputs_b = self.model(**inputs_b, output_hidden_states=True)

        embs_a = self._pooling(model_outputs_a, inputs_a.attention_mask, self.pool_type)
        embs_b = self._pooling(model_outputs_b, inputs_b.attention_mask, self.pool_type)  # batch_size x hidden_size

        sim_scores = []
        for i in range(self.persp_num):
            tmp_a = self.multi_persp_linears[i](embs_a)  # batch_size x hidden_size
            tmp_b = self.multi_persp_linears[i](embs_b)  # batch_size x hidden_size
            norm_a = torch.nn.functional.normalize(tmp_a, p=2, dim=1)
            norm_b = torch.nn.functional.normalize(tmp_b, p=2, dim=1)
            score = torch.sum(norm_a * norm_b, dim=1)    # batch_size
            # score = (score + 1) / 2  # convert score from (-1, 1) to (0, 1)
            score = score / self.temperature
            score = torch.exp(score) / (torch.exp(score) + torch.exp(-score))
            sim_scores.append(score)
        sim_scores = torch.stack(sim_scores, dim=0).transpose(0, 1)  # batch_size x persp_num

        predict_label = torch.gt(sim_scores, 0.5).int()
        if labels is None:
            loss = None
            acc  = None
        else:
            sim_scores_flatten = sim_scores.flatten()
            labels_flatten = labels.flatten().to(self.device)
            loss = self.loss(sim_scores_flatten, labels_flatten)
            acc = torch.mean((predict_label.flatten() == labels_flatten).float())

        return sim_scores, predict_label, loss, acc

    def save_model(self,
                   model_dir
        ):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)

        persp_linear_path = os.path.join(model_dir, 'persp_linear.bin')
        linear_state_dict = self.multi_persp_linears.state_dict()
        torch.save(linear_state_dict, persp_linear_path)
        return model_dir

    def load_model(self,
                   model_dir
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.to(self.device)

        persp_linear_path = os.path.join(model_dir, 'persp_linear.bin')
        linear_state_dict = torch.load(persp_linear_path, map_location=self.device)
        self.multi_persp_linears.load_state_dict(linear_state_dict, strict=True)
        return model_dir


def evaluate(args, model, eval_dataloader, sample_num):

    model.eval()
    correct_num_dict = {k: 0 for k in args.perspectives}
    sim_score_dict  = {k: [] for k in args.perspectives}
    true_label_dict = {k: [] for k in args.perspectives}
    pred_label_dict = {k: [] for k in args.perspectives}
    for sample_batch in tqdm(eval_dataloader, total=sample_num // args.batch_size, desc="Validating"):

        texta = sample_batch['texta']
        textb = sample_batch['textb']
        label = sample_batch['label'].to(args.device)

        with torch.no_grad():
            sim_scores, predict_label, loss, acc = model(texta, textb, label)

        for persp in args.perspectives:
            pidx = args.persp2idx[persp]
            persp_true_label = label[:, pidx].flatten()
            persp_pred_label = predict_label[:, pidx].flatten()
            correct_num_dict[persp] += torch.sum((persp_true_label == persp_pred_label).int()).item()

            persp_sim_score = sim_scores[:, pidx].flatten()
            sim_score_dict[persp] += persp_sim_score.tolist()
            true_label_dict[persp] += persp_true_label.int().tolist()
            pred_label_dict[persp] += persp_pred_label.int().tolist()

    # calculate auc
    auc_dict = {}
    acc_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    support_dict = {}
    for persp in args.perspectives:
        fpr, tpr, thresh = metrics.roc_curve(true_label_dict[persp], sim_score_dict[persp], pos_label=1)
        auc_value = metrics.auc(fpr, tpr)

        auc_dict[persp] = 0.0 if math.isnan(auc_value) else auc_value
        acc_dict[persp] = metrics.accuracy_score(true_label_dict[persp], pred_label_dict[persp])
        f1_dict[persp] = metrics.f1_score(true_label_dict[persp], pred_label_dict[persp], pos_label=1)
        precision_dict[persp] = metrics.precision_score(true_label_dict[persp], pred_label_dict[persp], pos_label=1)
        recall_dict[persp] = metrics.recall_score(true_label_dict[persp], pred_label_dict[persp], pos_label=1)
        support_dict[persp] = sum(true_label_dict[persp])

    # show evaluation result
    acc_sum, auc_sum = 0, 0
    precision_sum, recall_sum, f1_sum = 0, 0, 0
    support_sum = 0
    for persp in args.perspectives:
        support = support_dict[persp]
        acc_sum += acc_dict[persp] * support
        auc_sum += auc_dict[persp] * support
        precision_sum += precision_dict[persp] * support
        recall_sum += recall_dict[persp] * support
        f1_sum += f1_dict[persp] * support
        support_sum += support

        persp_fmt = persp.ljust(12, " ")
        print(f"{persp_fmt}: "
              f"acc={acc_dict[persp]:.4f}, "
              f"auc={auc_dict[persp]:.4f}, "
              f"precision={precision_dict[persp]:.4f}, "
              f"recall={recall_dict[persp]:.4f}, "
              f"f1={f1_dict[persp]:.4f}, "
              f"support={support_dict[persp]}")

    avg_acc = acc_sum / support_sum
    avg_auc = auc_sum / support_sum
    avg_precision = precision_sum / support_sum
    avg_recall = recall_sum / support_sum
    avg_f1 = f1_sum / support_sum
    print(f"W.Avg acc={avg_acc:.4f}, W.Avg auc={avg_auc:.4f}, "
          f"W.Avg precision={avg_precision:.4f}, W.Avg recall={avg_recall:.4f}, W.Avg f1={avg_f1:.4f}")

    return avg_acc, avg_auc, avg_precision, avg_recall, avg_f1


def get_optimizer_and_scheduler(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps   = math.ceil(args.total_training_steps * 0.1),
        num_training_steps = args.total_training_steps)
    return optimizer, scheduler


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['bi', 'cross'], required=True, help='bi: bi-encoder, cross: cross-encoder')
    parser.add_argument('--init_model_dir', type=str, required=True, help='Directory path of the model.')
    parser.add_argument('--save_model_dir', type=str, required=True, help='Directory path of output model.')
    parser.add_argument('--pool_type', type=str, default='avg', help='Pooling method.')
    parser.add_argument('--train_path', type=str, required=True, help='Path of the training dataset.')
    parser.add_argument('--valid_path', type=str, required=True, help='Path of the valid dataset.')
    parser.add_argument('--test_path', type=str, required=True, help='Path of the test dataset.')
    parser.add_argument('--max_length', type=int, default=128, help='Max seq length.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature of bi-encoder.')

    parser.add_argument("--num_epochs", type=int, default=10, help='The number of training epoches.')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size.')
    parser.add_argument("--learning_rate", type=float, default=5e-5, help='Learning rate.')
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help='Max value of gradient.')
    parser.add_argument("--verbose_per_step", type=int, default=100, help='Verbose per step.')
    parser.add_argument("--weight_decay", type=float, default=0.01, help='Weight decay rate.')
    parser.add_argument("--seed", type=int, default=7, help='Random seed.')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    train_dataset = MPTSDataset(args.train_path)
    valid_dataset = MPTSDataset(args.valid_path)
    test_dataset  = MPTSDataset(args.test_path)
    args.perspectives = train_dataset.perspectives
    args.persp_num = len(train_dataset.perspectives)
    args.persp2idx = train_dataset.persp2idx
    args.train_sample_num = len(train_dataset)
    args.valid_sample_num = len(valid_dataset)
    args.test_sample_num = len(test_dataset)
    print(f"There are {args.train_sample_num} training samples.")
    print(train_dataset.stastics())
    print(f"There are {args.valid_sample_num} valid samples.")
    print(valid_dataset.stastics())
    print(f"There are {args.test_sample_num} test samples.")
    print(test_dataset.stastics())

    args.steps_per_epoch = args.train_sample_num // args.batch_size
    args.total_training_steps = args.num_epochs * args.steps_per_epoch

    if args.mode == 'bi':
        print(f"Create bi-encoder from {args.init_model_dir}")
        model = BiEncoder(args)
    elif args.mode == 'cross':
        print(f"Create cross-encoder from {args.init_model_dir}")
        model = CrossEncoder(args)
    else:
        raise Exception("Unknown mode.")
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ImpsDataset.collate_fn_with_label)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ImpsDataset.collate_fn_with_label)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ImpsDataset.collate_fn_with_label)

    print("Start training.")
    best_f1 = 0
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}")
        loss_verbose, acc_verbose = 0, 0
        model.train()
        for step, sample_batch in enumerate(train_dataloader):

            texta_batch = sample_batch['texta']
            textb_batch = sample_batch['textb']
            label_batch = sample_batch['label'].to(args.device)

            sim_scores, predict_label, loss, acc = model(texta_batch, textb_batch, label_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_verbose += loss.item()
            acc_verbose += acc.item()
            if (step + 1) % args.verbose_per_step == 0:
                loss_verbose = loss_verbose / args.verbose_per_step
                acc_verbose = acc_verbose / args.verbose_per_step
                print(f"Epoch {epoch} step {step + 1} / {args.steps_per_epoch}: loss = {loss_verbose:.4f}, acc = {acc_verbose:.4f}")
                loss_verbose, acc_verbose = 0, 0

        print("Evaluate on valid dataset.")
        avg_acc, avg_auc, avg_precision, avg_recall, avg_f1 = evaluate(args, model, valid_dataloader, args.valid_sample_num)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            model.save_model(args.save_model_dir)
            print(f"Best model save to {args.save_model_dir}")
        print("\n")

    print(f"Loading best model from {args.save_model_dir}, and evaluate on test dataset")
    model.load_model(args.save_model_dir)
    evaluate(args, model, test_dataloader, args.test_sample_num)


if __name__ == "__main__":
    main()
