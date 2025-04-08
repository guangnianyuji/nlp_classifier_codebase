
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch.nn import CrossEntropyLoss
import pdb

class Bert(BertPreTrainedModel):

    def __init__(self, config):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        # self.fc = nn.Linear(self.hidden_size, self.num_classes)
        fc_dim = 64
        self.fc = nn.Sequential(*[
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, self.num_classes),
        ])

    def forward(self, 
                input_ids,
                attention_mask,
                label=None, 
                input_ids_anti=None, 
                label_anti=None):
        # inference  
        # pdb.set_trace()
        if input_ids.ndim == 3:
            input_ids = input_ids[:,0,:]

        output_bert = self.bert(input_ids, attention_mask=attention_mask)    #(batch_size, sen_length, hidden_size)
        output_pooler = output_bert.pooler_output
        # print(input_ids.shape, output_pooler.shape)
        output = self.fc(output_pooler)
        
        return [output, output_pooler]