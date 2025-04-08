# coding: UTF-8
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss

class XLMRobertaClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super(XLMRobertaClassifier, self).__init__(config)
        self.roberta = XLMRobertaModel(config)  # 使用 XLM-RoBERTa 作为基础模型
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        
        # 定义前馈神经网络
        fc_dim = 64
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, self.num_classes),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(input_ids, attention_mask=attention_mask)
        out = self.fc(output.pooler_output)
        return [out, output.pooler_output]

