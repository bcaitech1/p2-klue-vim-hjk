import pickle as pickle
import os
import pandas as pd
import torch
from pororo import Pororo
from tqdm import tqdm

ner = Pororo(task="ner", lang="ko")

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 'entity_02':dataset[5], 'label':label, 'entity_01_start':dataset[3], 'entity_01_end':dataset[4], 'entity_02_start':dataset[6], 'entity_02_end':dataset[7]})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512,
      add_special_tokens=True,
      )
  return tokenized_sentences


def roberta_tokenized_dataset(dataset, tokenizer):
  print('start processing...')
  for sent, e01, e02, start01, end01, start02, end02 in tqdm(zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02'], dataset['entity_01_start'], dataset['entity_01_end'], dataset['entity_02_start'], dataset['entity_02_end'])):
    ner_01 = ' ₩ ' + ner(e01)[0][1].lower() + ' ₩ '
    ner_02 = ' ∧ ' + ner(e02)[0][1].lower() + ' ∧ '

    entity_01_start, entity_01_end = int(start01), int(end01)
    entity_02_start, entity_02_end = int(start02), int(end02)

    if entity_01_start < entity_02_start:
      sent = sent[:entity_01_start] + '#' + ner_01 + sent[entity_01_start:entity_01_end + 1] + ' # ' + sent[entity_01_end + 1:entity_02_start] + \
             '@' + ner_02 + sent[entity_02_start:entity_02_end + 1] + ' @ ' + sent[entity_02_end + 1:]

  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '</s>' + '</s>' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences
