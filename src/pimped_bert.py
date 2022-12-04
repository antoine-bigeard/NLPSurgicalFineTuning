import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CombinationBlock(nn.Module):
    """For now just linear combination of the two blocks."""

    def __init__(self, block, frozen_block) -> None:
        super().__init__()
        self.block = block
        self.frozen_block = frozen_block
        self.alpha = nn.Parameter(torch.Tensor(0))

    def forward(self, x, extented_attention_mask):
        # return torch.sigmoid(self.alpha) * self.block(x) + (1 - torch.sigmoid(self.alpha)) * self.frozen_block(x)

        return (
            0.3 * self.block(x, attention_mask=extented_attention_mask)[0]
            + (1 - 0.3)
            * self.frozen_block(x, attention_mask=extented_attention_mask)[0]
        )


class SurgicalFineTuningBert(nn.Module):
    def __init__(
        self,
        bert_model,
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        # copy the model

        self.embedding_block = bert_model.bert.embeddings
        self.opti_bert_layers = bert_model.bert.encoder.layer
        self.frozen_bert_layers = copy.deepcopy(self.opti_bert_layers)
        self.opti_bert_pooler = bert_model.bert.pooler
        self.frozen_bert_pooler = copy.deepcopy(self.opti_bert_pooler)
        self.opti_bert_classifier = bert_model.classifier
        self.frozen_bert_classifier = copy.deepcopy(self.opti_bert_classifier)

        for param in self.frozen_bert_layers.parameters():
            param.requires_grad = False
        for param in self.frozen_bert_pooler.parameters():
            param.requires_grad = False
        for param in self.frozen_bert_classifier.parameters():
            param.requires_grad = False

        self.dropout = nn.Sequential(bert_model.dropout)
        self.alphas_layers = nn.Parameter(
            torch.zeros(len(bert_model.bert.encoder.layer))
        )
        self.alpha_pooler = nn.Parameter(torch.Tensor([0]))
        self.alpha_classifier = nn.Parameter(torch.Tensor([0]))
        # self.combination_blocks = nn.Sequential(
        #     *[
        #         CombinationBlock(
        #             self.bert_model.bert.encoder.layer[i],
        #             self.frozen_bert_model.bert.encoder.layer[i],
        #         )
        #         for i in range(len(self.bert_model.bert.encoder.layer))
        #     ]
        # )

    def forward(self, x):
        input_ids = x["input_ids"]
        x, attention_mask = (
            self.embedding_block(input_ids),
            x["attention_mask"],
        )
        extended_attention_mask = self.bert_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        for i in range(len(self.opti_bert_layers)):
            a = torch.sigmoid(self.alphas_layers[i])
            x = (
                a
                * self.opti_bert_layers[i](x, attention_mask=extended_attention_mask)[0]
                + (1 - a)
                * self.frozen_bert_layers[i](x, attention_mask=extended_attention_mask)[
                    0
                ]
            )
        a = torch.sigmoid(self.alpha_pooler)
        x = a * self.opti_bert_pooler(x) + (1 - a) * self.frozen_bert_pooler(x)
        x = self.dropout(x)

        a = torch.sigmoid(self.alpha_classifier)
        x = a * self.opti_bert_classifier(x) + (1 - a) * self.frozen_bert_classifier(x)

        return x
        # print("calling block forward")
        # x_input, attention_mask = (
        #     self.embedding_block(x["input_ids"]),
        #     x["attention_mask"],
        # )

        # extended_attention_mask = self.bert_model.get_extended_attention_mask(
        #     attention_mask, x["input_ids"].size()
        # )

        # # mask = x.attention_mask
        # return self.combination_blocks(x_input, extended_attention_mask)

    def get_alphas(self):
        return [block.alpha for block in self.combination_blocks]
