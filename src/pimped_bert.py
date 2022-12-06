import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from itertools import chain


class SurgicalFineTuningBert(nn.Module):
    def __init__(
        self,
        bert_model,
    ) -> None:
        super().__init__()
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        # copy the model

        self.opti_embedding_block = bert_model.bert.embeddings
        self.frozen_embedding_block = copy.deepcopy(self.opti_embedding_block)
        self.opti_bert_layers = bert_model.bert.encoder.layer
        self.frozen_bert_layers = copy.deepcopy(self.opti_bert_layers)
        self.opti_bert_pooler = bert_model.bert.pooler
        self.frozen_bert_pooler = copy.deepcopy(self.opti_bert_pooler)
        self.opti_bert_classifier = bert_model.classifier
        self.frozen_bert_classifier = copy.deepcopy(self.opti_bert_classifier)

        frozen_params = chain(
            self.frozen_embedding_block.parameters(),
            self.frozen_bert_layers.parameters(),
            self.frozen_bert_pooler.parameters(),
            self.frozen_bert_classifier.parameters(),
        )

        for param in frozen_params:
            param.requires_grad = False

        self.dropout = nn.Sequential(bert_model.dropout)
        self.alphas = torch.zeros(len(bert_model.bert.encoder.layer) + 1)
        # self.alphas_layers = nn.Parameter(
        #     torch.zeros(len(bert_model.bert.encoder.layer))
        # )
        # self.alpha_classifier = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        alphas_layers, alpha_classifier = self.alphas[:-1], self.alphas[-1]

        x_opti, x_frozen = self.opti_embedding_block(
            input_ids
        ), self.frozen_embedding_block(input_ids)

        for i in range(len(self.opti_bert_layers)):
            a = alphas_layers[i].sigmoid()
            if i > 0:
                x_opti, x_frozen = x, x
            x = (
                a
                * self.opti_bert_layers[i](
                    x_opti, attention_mask=extended_attention_mask
                )[0]
                + (1 - a)
                * self.frozen_bert_layers[i](
                    x_frozen, attention_mask=extended_attention_mask
                )[0]
            )

        x = self.frozen_bert_pooler(x)
        x = self.dropout(x)

        a = alpha_classifier.sigmoid()
        x = a * self.opti_bert_classifier(x) + (1 - a) * self.frozen_bert_classifier(x)

        return x

    def forward_alphas(self, x, alphas):
        alphas_layers, alpha_classifier = alphas[:-1], alphas[-1]

        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        x_opti, x_frozen = self.opti_embedding_block(
            input_ids
        ), self.frozen_embedding_block(input_ids)

        for i in range(len(self.opti_bert_layers)):
            a = alphas_layers[i]
            if i > 0:
                x_opti, x_frozen = x, x
            x = (
                a
                * self.opti_bert_layers[i](
                    x_opti, attention_mask=extended_attention_mask
                )[0]
                + (1 - a)
                * self.frozen_bert_layers[i](
                    x_frozen, attention_mask=extended_attention_mask
                )[0]
            )

        x = self.frozen_bert_pooler(x)
        x = self.dropout(x)

        a = alpha_classifier
        x = a * self.opti_bert_classifier(x) + (1 - a) * self.frozen_bert_classifier(x)

        return x

    def get_alphas(self):
        # return [round(self.sigmoid(float(a)), 4) for a in self.alphas]
        return [round(float(a.sigmoid()), 4) for a in self.alphas]

    def normalize_alphas(self, alpha_avg=0.5):
        self.alphas = (self.alphas.sigmoid() / self.alphas.sigmoid().sum()).logit()
        # n_alphas = 1 + len(self.alphas_layers)
        # total_sum_of_alphas_desired = n_alphas * alpha_avg

        # actual_sum_of_alphas = self.sigmoid(float(self.alpha_classifier))
        # for a in list(self.alphas_layers):
        #     actual_sum_of_alphas += self.sigmoid(float(a))

        # norm_factor = actual_sum_of_alphas/ total_sum_of_alphas_desired

        # with torch.no_grad():
        #     self.alpha_classifier = nn.Parameter(torch.logit(torch.sigmoid(self.alpha_classifier) / norm_factor))
        #     for i,a in enumerate(list(self.alphas_layers)):
        #         self.alphas_layers[i] = nn.Parameter(torch.logit(torch.sigmoid(a) / norm_factor))
