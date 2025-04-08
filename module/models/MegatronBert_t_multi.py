import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, MegatronBertModel,MegatronBertForSequenceClassification,AutoConfig,AutoTokenizer
import pdb

class MegatronBert_t_multi(MegatronBertForSequenceClassification):
    def __init__(self, config):
        super(MegatronBert_t_multi,self).__init__(config)
        print(">>>>>>>>>>>>>>>>>>.init MegatronBert...")
         
        config.output_hidden_states = True
        # print(config)

        self.MegatronBert = MegatronBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=config.initial_pretrain_model,
            config=config
        )

        print(">>>>>>>>>>>>>>>>>>.init MegatronBert over")
        self.hidden_size = config.hidden_size
        self.multi_num_classes = config.multi_num_classes
        # self.fc = nn.Linear(self.hidden_size, self.num_classes)
        fc_dim = 64
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, num_class),
            )
            for num_class in self.multi_num_classes
        ])

    def forward(self, 
                input_ids,
                attention_mask,
                label=None, 
                input_ids_anti=None, 
                label_anti=None):
        # inference  
        # if input_ids.ndim == 3:
        #     input_ids = input_ids[:,0,:]
        outputs = self.MegatronBert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.hidden_states[-1] 
        output_pooler=last_hidden[:, 0, :] 
        # print(output_pooler.shape)#torch.Size([6, 1536])
        outputs = [fc(output_pooler) for fc in self.fcs]#[torch.Size([6, 2]),torch.Size([6, 2]),torch.Size([6, 2]),torch.Size([6, 2]),torch.Size([6, 2])]
        
        return [outputs, output_pooler]




if __name__=="__main__":


    # 设置路径（你可以改成自己的模型路径）
    MODEL_PATH = "/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L"

    # # 模拟 config 对象（你项目中的 config 可能是自定义的）
    # class DummyConfig:
    #     def __init__(self):
    #         self.initial_pretrain_model = MODEL_PATH
    #         self.hidden_size = 1536  # 和你的 MegatronBERT 模型匹配
    #         self.multi_num_classes = [3, 2, 2, 2]  # 举个例子，你按你的模型写

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.initial_pretrain_model=MODEL_PATH
    config.output_hidden_states = True
    config.multi_num_classes = [3, 2, 2, 2]
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 创建模型实例
    model = MegatronBert_t_multi(config)
    model.eval()  # 切换到 eval 模式

    # 构造 dummy 输入（单条句子）
    text = "This is a test sentence for MegatronBert."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,max_length=1024)

    # 放到模型上跑 forward
    with torch.no_grad():
        outputs, pooled = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # 打印输出结果
    print("Logits output (list of tensors per task):")
    for i, out in enumerate(outputs):
        print(f"  Task {i}: shape = {out.shape}")  # [1, num_class_i]

    print("Sentence-level embedding shape:", pooled.shape)  # [1, hidden_size]
