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

    def forward(self, x, attention_mask):
        # return F.sigmoid(self.alpha) * self.block(x) + (1 - F.sigmoid(self.alpha)) * self.frozen_block(x)
        return 0.3 * self.block(x, attention_mask = attention_mask)[0] + (1 - 0.3) * self.frozen_block(x, attention_mask = attention_mask)[0]


class SurgicalFineTuningBert(nn.Module):
    def __init__(
        self,
        bert_model,
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        # copy the model
        self.frozen_bert_model = copy.deepcopy(bert_model)
        # freeze the model
        for param in self.frozen_bert_model.parameters():
            param.requires_grad = False

        self.embedding_block = bert_model.bert.embeddings
        self.combination_blocks = nn.Sequential(
            *[
                CombinationBlock(
                    self.bert_model.bert.encoder.layer[i],
                    self.frozen_bert_model.bert.encoder.layer[i],
                )
                for i in range(len(self.bert_model.bert.encoder.layer))
            ]
        )

    def forward(self, x):
        print("calling block forward")
        self.bert_model(**x)
        x_input, attention_mask = self.embedding_block(x['input_ids']), x['attention_mask']
        
        # mask = x.attention_mask
        return self.combination_blocks(x_input, attention_mask)

    def get_alphas(self):
        return [block.alpha for block in self.combination_blocks]

