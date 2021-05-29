import os
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from easydict import EasyDict
from prettyprinter import cpprint
from importlib import import_module
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from madgrad import MADGRAD
from adamp import AdamP


from load_data import *
from bert_relation_classification import BertForSequenceClassification


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Config
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()        
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()
    
    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()
    
    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.3, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = LabelSmoothingLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

# 시드 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': round(acc, 4)
    }

def train(cfg):
    SEED = cfg.values.seed
    MODEL_NAME = cfg.values.model_name
    USE_KFOLD = cfg.values.val_args.use_kfold
    TRAIN_ONLY = cfg.values.train_only

    seed_everything(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model_config_module = getattr(import_module('transformers'), cfg.values.model_arc + 'Config')
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42

    whole_df = load_data("/opt/ml/input/data/train/train.tsv")
    additional_df = load_data("/opt/ml/input/data/train/additional_train.tsv")

    whole_label = whole_df['label'].values
    # additional_label = additional_df['label'].values

    if cfg.values.tokenizer_arc:
        tokenizer_module = getattr(import_module('transformers'), cfg.values.tokenizer_arc)
        tokenizer = tokenizer_module.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=9999999, early_stopping_threshold=0.001)

    training_args = TrainingArguments(
            output_dir=cfg.values.train_args.output_dir,          # output directory
            save_total_limit=cfg.values.train_args.save_total_limit,              # number of total save model.
            save_steps=cfg.values.train_args.save_steps,                 # model saving step.
            num_train_epochs=cfg.values.train_args.num_epochs,              # total number of training epochs
            learning_rate=cfg.values.train_args.lr,               # learning_rate
            per_device_train_batch_size=cfg.values.train_args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=cfg.values.train_args.eval_batch_size,   # batch size for evaluation         
            warmup_steps=cfg.values.train_args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=cfg.values.train_args.weight_decay,               # strength of weight decay            
            max_grad_norm=cfg.values.train_args.max_grad_norm,
            logging_dir=cfg.values.train_args.logging_dir,            # directory for storing logs
            logging_steps=cfg.values.train_args.logging_steps,              # log saving step.
            evaluation_strategy=cfg.values.train_args.evaluation_strategy, # evaluation strategy to adopt during training
                                        # `no`: No evaluation during training.
                                        # `steps`: Evaluate every `eval_steps`.
                                        # `epoch`: Evaluate every end of epoch.
            eval_steps=cfg.values.train_args.eval_steps,            # evaluation step.
            dataloader_num_workers=4, 
            seed=SEED,
            label_smoothing_factor=cfg.values.train_args.label_smoothing_factor,
            load_best_model_at_end=True,
            # metric_for_best_model='accuracy'
            )
    
    if USE_KFOLD:
        kfold = StratifiedKFold(n_splits=cfg.values.val_args.num_k)

        k = 1
        for train_idx, val_idx in kfold.split(whole_df, whole_label):
            print('\n')
            cpprint('=' * 15 + f'{k}-Fold Cross Validation' + '=' * 15)
            train_df = whole_df.iloc[train_idx]
            # train_df = pd.concat((train_df, additional_df))
            val_df = whole_df.iloc[val_idx]

            if cfg.values.model_arc == 'Roberta':
                tokenized_train = roberta_tokenized_dataset(train_df, tokenizer)
                tokenized_val = roberta_tokenized_dataset(val_df, tokenizer)
            else:
                tokenized_train = tokenized_dataset(train_df, tokenizer)
                tokenized_val = tokenized_dataset(val_df, tokenizer)   

            RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
            RE_val_dataset = RE_Dataset(tokenized_val, val_df['label'].values)

            
            try:
                if cfg.values.model_name == 'Bert':
                    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            except:
                # model_module = getattr(import_module('transformers'), cfg.values.model_arc)
                model_module = getattr(import_module('transformers'), cfg.values.model_arc + 'ForSequenceClassification')
                model = model_module.from_pretrained(MODEL_NAME, config=model_config)
            
            model.parameters
            model.to(device)
            
            training_args.output_dir = cfg.values.train_args.output_dir + f'/{k}fold'
            training_args.logging_dir = cfg.values.train_args.output_dir + f'/{k}fold'

            optimizer = MADGRAD(model.parameters(), lr=training_args.learning_rate)
            total_step = len(RE_train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_step)
            optimizers = optimizer, scheduler

            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_val_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                optimizers=optimizers,
                # callbacks=[early_stopping]
            )
            k += 1
            # train model
            trainer.train()

    else:
        cpprint('=' * 20 + f'START TRAINING' + '=' * 20)
        if not TRAIN_ONLY:
            train_df, val_df= train_test_split(whole_df, test_size=cfg.values.val_args.test_size, random_state=SEED)
            # train_df = pd.concat((train_df, additional_df))

            if cfg.values.model_arc == 'Roberta':
                tokenized_train = roberta_tokenized_dataset(train_df, tokenizer)
                tokenized_val = roberta_tokenized_dataset(val_df, tokenizer)
            else:
                tokenized_train = tokenized_dataset(train_df, tokenizer)
                tokenized_val = tokenized_dataset(val_df, tokenizer) 

            RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
            RE_val_dataset = RE_Dataset(tokenized_val, val_df['label'].values)

            try:
                if cfg.values.model_name == 'Bert':
                    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            except:
                # model_module = getattr(import_module('transformers'), cfg.values.model_arc)
                model_module = getattr(import_module('transformers'), cfg.values.model_arc + 'ForSequenceClassification')
                model = model_module.from_pretrained(MODEL_NAME, config=model_config)
            
            model.parameters
            model.to(device)            
            
            optimizer = transformers.AdamW(model.parameters(), lr=training_args.learning_rate)
            total_step = len(RE_train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs
            # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_step)           
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_step)
            optimizers = optimizer, scheduler

            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_val_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                optimizers=optimizers,
                callbacks=[early_stopping]
            )
            
            # train model
            trainer.train()
        
        else:            
            training_args.evaluation_strategy = 'no'

            if cfg.values.model_arc == 'Roberta':
                print('Roberta')
                tokenized_train = roberta_tokenized_dataset(whole_df, tokenizer)
            else:
                tokenized_train = tokenized_dataset(whole_df, tokenizer)

            

            RE_train_dataset = RE_Dataset(tokenized_train, whole_df['label'].values)

            try:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            except:
                # model_module = getattr(import_module('transformers'), cfg.values.model_arc)
                model_module = getattr(import_module('transformers'), cfg.values.model_arc + 'ForSequenceClassification')
                model = model_module.from_pretrained(MODEL_NAME, config=model_config)
            
            model.parameters
            model.to(device)

            training_args.output_dir = cfg.values.train_args.output_dir + '/only_train'
            training_args.logging_dir = cfg.values.train_args.output_dir + '/only_train'
            
            optimizer = AdamP(model.parameters(), lr=training_args.learning_rate)
            total_step = len(RE_train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_step)
            optimizers = optimizer, scheduler

            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                optimizers=optimizers,
                # callbacks=[early_stopping]
            )
            
            # train model
            trainer.train()

def main(cfg):
    train(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='base')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)