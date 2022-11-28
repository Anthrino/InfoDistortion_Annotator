import torch
import csv
import pprint
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModel, AutoConfig

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from datasets import load_metric
from tqdm.auto import tqdm
import os
import random
import pickle


def load_dataset(filename):

    headline_pairs = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        headline_tuples = list(csvreader)
    # for i in headline_tuples:
    #     headline_pairs.append(((i[0], i[1]), i[2]))

    return headline_tuples


def tokenize(batch):
    return tokenizer(batch, truncation=True, max_length=100)


class BERT():

    def __init__(self, model_name="bert-base-uncased") -> None:
        #self.model = BertModel.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=1)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(768, 1)
        self.loss_fct = nn.CrossEntropyLoss()

    def get_tokenizer_and_model(self):
        return self.model, self.tokenizer

    # def tokenize_data(self, data):
    #     return self.tokenizer.tokenize(data)

    def forward(self, input_ids, attn_mask, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attn_mask, labels=labels)
        return outputs

        # sequence_outputs = self.dropout(outputs[1])
        # logits = torch.sigmoid(self.linear(sequence_outputs[:, 0, :].view(-1, 768)))
        # logits = outputs.logits
        # loss = None

        # if labels is not None:
        #     # loss = self.loss_fct(logits.view(-1, 1), labels)

        #     return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        # extract the 1st token's embeddings


if __name__ == "__main__":

    bert = BERT()
    model, tokenizer = bert.get_tokenizer_and_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    headline_pairs_pos = load_dataset('dataset/pos_headline_pairs.tsv')
    headline_pairs_neg = load_dataset('dataset/neg_headline_pairs_pp0.2.tsv')
    train_size = int(0.8 * len(headline_pairs_pos))
    test_size = len(headline_pairs_pos) - train_size

    train_dataset_pos, test_dataset_pos = torch.utils.data.random_split(
        headline_pairs_pos, [train_size, test_size])

    train_size = int(0.8 * len(headline_pairs_neg))
    test_size = len(headline_pairs_neg) - train_size

    train_dataset_neg, test_dataset_neg = torch.utils.data.random_split(
        headline_pairs_neg, [train_size, test_size])

    train_dataset = list(train_dataset_neg) + list(train_dataset_pos)
    test_dataset = test_dataset_neg + test_dataset_pos

    random.shuffle(train_dataset)
    # train_dataloader = DataLoader(train_dataset, shuffle=True)

    # train_dataset = load_dataset('')
    # validation_dataset = load_dataset('')
    # test_dataset = load_dataset('')

    # encoding_train = bert.tokenize_data(train_dataset)
    # encoding_test = bert.tokenize_data(test_dataset)
    # encoding_validation = bert.tokenize_data(validation_dataset)

    epochs = 3
    batch_size = 16
    progress_bar_train = tqdm(range(epochs * len(train_dataset)))
    progress_bar_test = tqdm(range(epochs * len(test_dataset)))

    # softmax_fct = nn.Softmax()
    sigmoid_fct = nn.Sigmoid()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    print("\n Initiating Model Training..", end='\n\n')
    model.train()

    for epoch in range(epochs):
        for batch_idx in range(0, len(train_dataset), batch_size):

            optimizer.zero_grad()

            batch = train_dataset[batch_idx:batch_idx+batch_size]

            seq_pairs, targets = [], []
            for pair in batch:
                targets.append(np.float32(pair[2]))
                seq_pairs.append((pair[0], pair[1]))

            targets = torch.from_numpy(np.array(targets))

            # encoding = bert.tokenizer(pair[0], pair[1], return_tensors='pt', padding="max_length", truncation=True, max_length=100, add_special_tokens=True)
            encoding = bert.tokenizer.batch_encode_plus(
                seq_pairs, return_tensors='pt', padding=True, truncation=True, max_length=100, add_special_tokens=True)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # outputs = torch.nn.functional.log_softmax(outputs, dim=1)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=targets)

            # print(sigmoid_fct(outputs.logits), targets, outputs.loss)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar_train.update(batch_size)

    predictions = []
    labels = []

    eval_metric_accuracy = load_metric("accuracy")
    eval_metric_f1 = load_metric("f1")
    eval_metric_recall = load_metric("recall")
    eval_metric_precision = load_metric("precision")

    print("\n Initiating Model Evaluation..", end='\n\n')
    model.eval()

    with open('experiments/results/test_headline_predictions.tsv', 'w') as pred_file:
        
        # creating csv writer objects for dataset samples
        pred_file_writer = csv.writer(pred_file, delimiter="\t")
        for pair in test_dataset:
            encoding = bert.tokenizer(pair[0], pair[1], return_tensors='pt', padding="max_length",
                                    truncation=True, max_length=100, add_special_tokens=True)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            with torch.no_grad():
                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=targets)

            predict_score = sigmoid_fct(outputs.logits)
            out_label = '0' if predict_score < 0.5 else '1'

            predictions.append((predict_score, out_label))
            labels.append(pair[2])
            print(predict_score.item(), out_label, pair[2])
            pred_file_writer.writerow([pair, predict_score.item(), out_label, pair[2]])

            eval_metric_accuracy.add(predictions=out_label, references=pair[2])
            eval_metric_f1.add(predictions=out_label, references=pair[2])
            eval_metric_recall.add(
                predictions=out_label, references=pair[2])
            eval_metric_precision.add(
                predictions=out_label, references=pair[2])
        # progress_bar_test.update(1)

    print()
    print(eval_metric_accuracy.compute())
    print(eval_metric_precision.compute(average=None, zero_division=0))
    print(eval_metric_recall.compute(average=None, zero_division=0))
    print(eval_metric_f1.compute(average=None))

    with open('dataset/predictions2.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    with open('dataset/labels2.pkl', 'wb') as f:
        pickle.dump(labels, f)
