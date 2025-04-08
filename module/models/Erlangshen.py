from transformers import AutoModelForSequenceClassification, BertPreTrainedModel, BertForSequenceClassification
import torch
import torch.nn as nn
import pdb
class Erlangshen(BertPreTrainedModel):
    
    def __init__(self, config, initial_pretrain_model, num_labels):
        super(Erlangshen, self).__init__(config)
        self.Erlangshen = AutoModelForSequenceClassification.from_pretrained(initial_pretrain_model)
        self.num_classes = num_labels
        if not self.Erlangshen.classifier.out_features == num_labels:
            print('The classifier layer reinitialized')
            self.Erlangshen.classifier = torch.nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

    def forward(self, 
                input_ids,
                attention_mask,
                label=None, 
                input_ids_anti=None, 
                label_anti=None):

        if input_ids.ndim == 3:
            input_ids = input_ids[:,0,:]
        output = self.Erlangshen(input_ids)    #(batch_size, num_classes)
        return [output.logits, None]
