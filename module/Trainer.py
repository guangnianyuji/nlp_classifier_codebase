
import os
from posixpath import sep
import pdb
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
from safetensors.torch import load_file
import json
#from apex import amp
# from tqdm.auto import tqdm
from tqdm import tqdm
# from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, AutoConfig
# from model.BertForMaskedLM import BertForMaskedLM
import logging

import torch.nn.functional as F
from sklearn import metrics
import itertools
# from utils.progressbar import ProgressBar
from module.optimal.adversarial import FGM,PGD
from module.ModelMap import map_model, map_config, map_tokenizer
from module.LossManager import LossManager
from module.textsplit import TextSplitter

from safetensors.torch import load_file
import torch.distributed as dist
def repeat_and_concat(samples, num_list):
    result = []
    for sample, num in zip(samples, num_list):
        result.extend([sample] * num)

    if isinstance(samples, torch.Tensor):
        return torch.stack(result, dim=0)
    return result


from sklearn.utils.class_weight import compute_class_weight
def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)
def get_pr_ori(scores,labels):
    precision_thresholds = [0.9, 0.8, 0.7, 0.6,0.5,0.4,0.3,0.2,0.1]

    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]  # 使用排序的索引
    sorted_scores = scores[sorted_indices]  # 使用排序的索引

    recall_at_precision = {}
    total_selected = 0
    total_correct = 0
    score_thresh_map = {p: None for p in precision_thresholds}

    for j in range(len(sorted_scores)):
        total_selected += 1
        total_correct += sorted_labels[j]

        precision = total_correct / total_selected

        # 记录 Precision ≥ p 的最小分数阈值
        for p in precision_thresholds:
            if precision >= p :
                score_thresh_map[p] = sorted_scores[j]

    # 计算 Recall 只用一次遍历
    for p in precision_thresholds:
        score_thresh = score_thresh_map[p]
        if score_thresh is not None:
            selected_labels = sorted_labels[sorted_scores >= score_thresh]
            recall = np.sum(selected_labels) / np.sum(labels)
        else:
            recall = 0.0  # 没有满足 Precision 的阈值

        recall_at_precision[f"P@{int(p*100)}"] = {"score_thresh": score_thresh, "recall": recall}
    return recall_at_precision
def load_state(file_path):
    """
    根据文件格式自动加载模型权重：
    - 如果是 `.safetensors`，使用 `safetensors.torch.load_file()`
    - 如果是 `.bin` 或 `.pt`，使用 `torch.load()`
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 文件 `{file_path}` 不存在！")

    if file_path.endswith(".safetensors"):
        print(f"🔍 发现 Safetensors 文件 `{file_path}`，使用 `safetensors` 加载")
        state_dict = load_file(file_path)  # 加载 safetensors 格式
    elif file_path.endswith(".bin") or file_path.endswith(".pt"):
        print(f"🔍 发现 PyTorch Checkpoint `{file_path}`，使用 `torch.load()` 加载")
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))  # 加载 PyTorch 权重
    elif file_path.endswith(".safetensors.index.json"):
        print(f"🔍 发现 Sharded Safetensors 模型索引 `{file_path}`，使用索引加载所有分片")
        
        # 读取索引文件
        with open(file_path, "r") as f:
            index_data = json.load(f)
        
        # 提取所有分片路径（相对于索引文件）
        weight_files = index_data.get("weight_map", {}).values()
        base_dir = os.path.dirname(file_path)
        shard_paths = [os.path.join(base_dir, fname) for fname in set(weight_files)]

        # 加载所有分片
        state_dict = {}
        for shard in shard_paths:
            print(f"📦 正在加载分片 `{shard}`")
            state_dict.update(load_file(shard))  # safetensors 加载，每个是 dict
    else:
        raise ValueError(f"❌ 不支持的文件格式 `{file_path}`，请提供 `.safetensors` 或 `.bin/.pt`")

    return state_dict
class Trainer(object):
    
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.config = config
        if not os.path.exists(self.config.path_model_save):
            os.makedirs(self.config.path_model_save)
        # logging.basicConfig(filename=os.path.join(self.config.path_model_save,'log.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # 设置GPU环境
        self.local_rank=0

        if self.config.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            # init_distributed_mode(config)
            # print('>>>>> DDP initialized <<<<<<')
            self.device = torch.device(f'cuda:{local_rank}')
            self.config.path_model_save_rank=os.path.join(self.config.path_model_save,str(local_rank))
            os.makedirs(self.config.path_model_save_rank, exist_ok=True)
            self.local_rank=local_rank
        else:
            self.device = torch.device(self.config.device)
        # self.device = torch.device(self.config.device)
        # 加载数据集
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        # 加载标签
        # self.load_label()
        # 加载模型
        
        self.tokenizer=config.tokenizer
        self.load_model()
        # 加载loss计算类


        # 代码补全
        weights = None
        if self.config.mode == 'train' and self.config.use_loss_weight:
            labels = self.train_loader.dataset.pd_dataset['label'].values  # 提取所有标签
            weights = compute_class_weights(labels)  # 计算类别权重
            weights = torch.tensor(weights, dtype=torch.float32)

            print("self.device",self.device)
            # 计算类别权重并转换为 tensor
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)  # 确保在 self.device 上

            print("use_loss_weight: ")
            print(weights)
            print(f"weights device: {weights.device}")

        self.loss_manager = LossManager(loss_type=config.loss_type, cl_option=config.cl_option, loss_cl_type=config.cl_method,weights=weights)
        self.text_splitter = TextSplitter()
        



    def load_label(self):
        """
        读取标签
        """
        path_label = self.config.class_file_path
        self.label = [ x.strip() for x in open(path_label, 'r', encoding='utf8').readlines()]
        self.label2ids = {x:i for i,x in enumerate(self.label)}
        self.ids2label = {i:x for i,x in enumerate(self.label)}



    def load_model(self):
        """
        加载模型及初始化模型参数
        """
        # 读取模型
        print('loading model...%s' %self.config.model_name)
        self.model = map_model(self.config.model_name)
        if not self.model:
            print('model {} is null, please check your model name.'.format(self.config.model_name))
        
        if self.config.model_name not in self.config.lm_model_list:
            # self.model = map_model(self.config.model_name)
            model_config = map_config(self.config.model_name)(self.config)
            self.model = self.model(model_config)
            # 重新初始化模型参数
            self.init_network()
        elif self.config.model_name == 'Erlangshen':
            # pdb.set_trace() 
            # model_config = AutoConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=len(self.label))   #, num_labels=len(self.label2ids)
            # self.model = self.model(config=model_config, initial_pretrain_model=self.config.initial_pretrain_model,num_labels=len(self.label)) 
            self.model = self.model.from_pretrained(self.config.initial_pretrain_model)
            if not self.model.classifier.out_features == self.config.num_classes:
                print('The classifier layer reinitialized')
                self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=self.config.num_classes, bias=True)
            if len(self.config.resume_from) > 0:
                print(f'resume from {self.config.resume_from}')
                path_model = os.path.join(self.config.resume_from, 'pytorch_model.bin')
                self.model.load_state_dict(torch.load(path_model))
        # elif self.config.model_name == 'MegatronBert_t':
        #     # pdb.set_trace()
        #     self.model = self.model.from_pretrained(self.config.initial_pretrain_model)
        #     # self.model.bert.embeddings.position_embeddings = nn.Embedding(400, 1536)
        #     if not self.model.classifier.out_features == self.config.num_classes:
        #         print('The classifier layer reinitialized')
        #         self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=self.config.num_classes, bias=True)
        #     if len(self.config.resume_from) > 0:
        #         print(f'resume from {self.config.resume_from}')
        #         path_model = os.path.join(self.config.resume_from, 'pytorch_model.bin')
        #         self.model.load_state_dict(torch.load(path_model))
        #         # 读取配置文件（如果需要使用配置初始化模型）
        #         # from safetensors.torch import load_file
        #         # index_path = os.path.join(self.config.resume_from, 'model.safetensors.index.json')
        #         # with open(index_path, 'r') as f:
        #         #     index = json.load(f)

        #         # for part in index['weight_map'].values():
        #         #     file_path = os.path.join(self.config.resume_from, part)
        #         #     weights = load_file(file_path)
        #         #     self.model.load_state_dict(weights, strict=False)
        #     if not self.model.classifier.out_features == self.config.num_classes:
        #         print('The classifier layer reinitialized')
        #         self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=self.config.num_classes, bias=True)
        elif self.config.model_name == 'MegatronBert_t_multi':
            model_config = AutoConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=self.config.num_classes)
            model_config.initial_pretrain_model=self.config.initial_pretrain_model
            model_config.multi_num_classes=self.config.multi_num_classes
            self.model=self.model(model_config)
        else:
            # self.tokenizer = map_tokenizer(self.config.model_name).from_pretrained(self.config.model_pretrain_online_checkpoint)
            # self.tokenizer.save_pretrained(self.config.path_tokenizer)
            # self.func_index2token = self.tokenizer.convert_ids_to_tokens
            # 加载预训练模型
            if self.config.resume_from != '':
                self.model = self.model.from_pretrained(self.config.initial_pretrain_model,num_labels=self.config.num_classes)
                print(f">>>>>>>>>>>>>>load initial model from {self.config.initial_pretrain_model}")
                # model_config = AutoConfig.from_pretrained(self.config.resume_from, num_labels=2)   #, num_labels=len(self.label2ids)
                # self.model = self.model.from_pretrained(self.config.resume_from, config=model_config)  
                # if 
                # if not self.model.fc[-1].out_features == self.config.num_classes:
                #     print('The classifier layer reinitialized')
                #     self.model.fc[-1] = torch.nn.Linear(in_features=self.model.fc[-1].in_features, out_features=self.config.num_classes, bias=True)

                msg=self.model.load_state_dict(load_state(self.config.resume_from))
                print(f">>>>>>>>>>>>>>>>>>load weight from {self.config.resume_from}, with msg :{msg}") 
            else:
                model_config = AutoConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=self.config.num_classes)   #, num_labels=len(self.label2ids)
                model_config.initial_pretrain_model=self.config.initial_pretrain_model
                model_config.multi_num_classes=self.config.multi_num_classes
                self.model = self.model.from_pretrained(self.config.initial_pretrain_model, config=model_config,attn_implementation="eager")  

        # 将模型加载到CPU/GPU 
        if self.config.distributed:
            self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        else:

            if torch.cuda.device_count() > 1 and ',' in self.config.cuda_visible_devices:
                self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])
       

            self.model.to(self.device)

    
    def init_network(self, method='xavier', exclude='embedding', seed=123):
        """
        # 权重初始化，默认xavier
        """
        for name, w in self.model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        if 'transformer' in name:
                            nn.init.uniform_(w, -0.1, 0.1)
                        else:
                            nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass


    def train(self):
        """
            预训练模型
        """
        # weight decay
        # bert_parameters = self.model.bert.named_parameters()
        # start_parameters = self.model.start_fc.named_parameters()
        # end_parameters = self.model.end_fc.named_parameters()
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': self.config.learning_rate},
        #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': self.config.learning_rate},
        #     {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': 0.001},
        #     {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001}]
        # step_total = self.config.num_epochs * len(self.train_loader) * self.config.batch_size
        # # step_total = 640 #len(train_ld)*config.batch_size // config.num_epochs
        # warmup_steps = int(step_total * self.config.num_warmup_steps)
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=1e-8)
        # self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=step_total)
        
        # 定义优化器配置
        # num_training_steps = self.config.num_epochs * len(self.train_loader)
        # 总的训练次数
        step_total = self.config.num_epochs * len(self.train_loader) * self.config.batch_size
        # warm up的次数
        warmup_steps = int(step_total * self.config.num_warmup_steps)
        if self.config.model_name not in self.config.lm_model_list:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=step_total
            )
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
            #                                             num_training_steps=t_total)
        
        # 混合精度训练
        if self.config.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.config.fp16_opt_level)
        # 分布式训练
        
        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm = FGM(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)
        if self.config.adv_option == 'PGD':
            self.pgd = PGD(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)

        # Train!
        print("\n>>>>>>>> Running training >>>>>>>>")
        print("  Num examples = %d" %(len(self.train_loader)*self.config.batch_size))
        print("  Num Epochs = %d" %self.config.num_epochs)
        print("  Batch size per GPU = %d"%self.config.batch_size)
        print("  GPU ids = %s" %self.config.cuda_visible_devices)
        print("  Total step = %d" %step_total)
        print("  Warm up step = %d" %warmup_steps)
        print("  FP16 Option = %s" %self.config.fp16)
        print(">>>>>>>> Running training >>>>>>>>\n")
        
        """print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")"""

        # step_total = config.num_epochs * len(train_ld)
        step_current = 0
        f1_best = 0


        f1_eval = self.evaluate(self.valid_loader,print_table=True,step=0)
        # # 模型保存
        f1_best = self.save_checkpoint(step_current, f1_eval, f1_best)


        for epoch in range(self.config.num_epochs):
            # progress_bar = ProgressBar(n_total=len(self.train_loader), desc='Training epoch:{0}'.format(epoch))
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
            for i, batch in pbar:  # 这里 `pbar` 是 `tqdm` 实例

                # 模型推断及计算损失
                self.model.train()
                # pdb.set_trace()
                loss = self.step(batch, i)
                pbar.set_postfix({'loss': loss.item()})  # 在进度条后显示结果 
                # progress_bar(i, {'loss': loss.item()})
                # progress_bar(i, {'loss': loss.item(),'loss_ce': loss_ce.item(),'loss_cl': loss_nce.item()})
                step_current += 1
                # 模型保存
                if step_current%self.config.step_save==0 and step_current>0:
                    # 模型评估
                    f1_eval = self.evaluate(self.valid_loader,print_table=True,step=step_current)
                    # 模型保存
                    if self.local_rank==0:
                        f1_best = self.save_checkpoint(step_current, f1_eval, f1_best)
            print('\nEpoch:{}  Iter:{}/{}  loss:{:.4f}\n'.format(epoch, step_current, step_total, loss.item()))
        if not self.test_loader is None:
            self.evaluate(self.test_loader, print_table=True)
    
    

    def step(self, batch, epoch):
        """
        每一个batch的训练过程
        """
        # 数据操作
        if self.config.slide_window:
            split_content_list = [self.text_splitter(c) for c in batch['text']]
            num_split_content_list = [len(x) for x in split_content_list]
            split_content_list = list(itertools.chain.from_iterable(split_content_list))

            if self.config.add_cate:
                text = self.train_loader.dataset._text_to_encoding(split_content_list,repeat_and_concat([a + b for a, b in zip(batch['cate'], batch['cate2'])],num_split_content_list))
            else:
                text = self.train_loader.dataset._text_to_encoding(split_content_list, cate=None)
            input_ids = torch.LongTensor(text['input_ids']).to(self.device)
            attention_mask = text['attention_mask'].to(self.device)
        else:
            input_ids = batch['text']['input_ids'].to(self.device)
            attention_mask  = batch['text']['attention_mask'].to(self.device)

        target = batch['label'].to(self.device)
        if 'label1' in batch.keys():
            target1 = batch['label1'].to(self.device)
            target2 = batch['label2'].to(self.device)
            target3 = batch['label3'].to(self.device)
        # print(input_ids.shape, target.shape)

        # 模型输入&输出
        if self.config.model_name == 'MegatronBert_t' or self.config.model_name == 'Erlangshen':
            output = self.model(input_ids, attention_mask=attention_mask).logits
        else:
            outputs = self.model(input_ids, attention_mask)
            output, hidden_emb = outputs
        # print("="*80)
        # print(output)

        idx_list = batch['idx']

        if self.config.slide_window:
            # 低俗
            # if self.config.multi_num_classes != []:
            #     output1, output2 = output[:, :2], output[:, 2:] 
            #     output1, output2 = output1.split(num_split_content_list), output2.split(num_split_content_list)
            #     output1, output2 = torch.stack([x.max(dim=0)[0] for x in output1], dim=0), torch.stack([x.max(dim=0)[0] for x in output2], dim=0)
            #     output = torch.cat((output1, output2), dim=1)
            # if self.config.multi_num_classes != []:
            #     output0, output1, output2, output3 = output[:, :3], output[:, 3:5], output[:, 5:7], output[:, 7:9] 

            #     output0 = output0.split(num_split_content_list)
            #     output0 = torch.stack([x.max(dim=0)[0] for x in output0], dim=0)

            #     output1 = output1.split(num_split_content_list)
            #     output1 = torch.stack([x.max(dim=0)[0] for x in output1], dim=0)

            #     output2 = output2.split(num_split_content_list)
            #     output2 = torch.stack([x.max(dim=0)[0] for x in output2], dim=0)

            #     output3 = output3.split(num_split_content_list)
            #     output3 = torch.stack([x.max(dim=0)[0] for x in output3], dim=0)

            #     output = torch.cat((output0, output1, output2, output3), dim=1) 
            if self.config.multi_num_classes != []:
                output_list = []
                end = 0
                for i,num_classes in enumerate(self.config.multi_num_classes):
                    output0 = output[i]  # 当前子任务的 logits
                    output0 = output0.split(num_split_content_list)  # 按组切
                    # 对每组取 max-pooling（dim=0 表示在 chunk 维度上）
                    output0 = torch.stack([x.max(dim=0)[0] for x in output0], dim=0)
                    output_list.append(output0)
                    end += num_classes

                output = torch.cat(output_list, dim=1)

            else:
                output = output.split(num_split_content_list)
                output = torch.stack([x.max(dim=0)[0] for x in output], dim=0)

        # 对比学习
        if self.config.cl_option:
            # 重新获取一次模型输出
            outputs_etx = self.model(input_ids, attention_mask)
            _, hidden_emb_etx = outputs_etx
            loss = self.loss_manager.compute(output, target, hidden_emb, hidden_emb_etx, alpha=self.config.cl_loss_weight)
        else:
            if self.config.multi_num_classes != []:
                # loss0 = self.loss_manager.compute(output0, target)
                # loss1 = self.loss_manager.compute(output1, target1)
                # loss2 = self.loss_manager.compute(output2, target2)
                # loss3 = self.loss_manager.compute(output3, target3)
                # loss = loss0 + loss1 + loss2 + loss3
                end = 0
                loss=None
                # print(target)
                for i,num_classes in enumerate(self.config.multi_num_classes):
                    target0=target[:,i:i+1]#[4,1]
                    output0 = output_list[i] #[4,2]
                    output0 = output0[(target0 != -1).squeeze()]
                    target0=target0[target0!=-1]
                    if(target0.shape[0]==0):
                        continue

                    end += num_classes
                    loss0 = self.loss_manager.compute(output0, target0)
                    loss = loss0 if loss is None else loss + loss0
                # print("loss",loss)


            elif self.config.loss_type == 'FocalLossWithDenoise':
                loss = self.loss_manager.compute(output, target, idx_list, epoch)
            else:
                loss = self.loss_manager.compute(output, target)
        # 反向传播
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # 对抗训练
        self.attack_train(batch)
        # 梯度操作
        self.optimizer.step()
        if self.config.model_name in self.config.lm_model_list:
            self.lr_scheduler.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        
        return loss


    def attack_train(self, batch):
        """
        对抗训练
        """
        # FGM
        if self.config.adv_option == 'FGM':
            self.fgm.attack()
            output = self.model(**batch)[0]
            loss_adv = self.loss_manager.compute(output, batch['label'])
            if torch.cuda.device_count() > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.fgm.restore()
        # PGD
        if self.config.adv_option == 'PGD':
            self.pgd.backup_grad()
            K = 3
            for t in range(K):
                self.pgd.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                output = self.model(**batch)[0]
                loss_adv = self.loss_manager.compute(output, batch['label'])
                loss_adv.backward()                      # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd.restore()   
            

    def save_checkpoint(self, step_current, f1_eval, f1_best):
        """
        模型保存
        """
        if f1_eval != 0:
            # 保存路径
            """path = os.path.join(self.config.path_model_save, 'step_{}'.format(step_current))
            if not os.path.exists(path):
                os.makedirs(path)
            # 保存当前step的模型
            if self.config.model_name not in self.config.lm_model_list:
                path_model = os.path.join(path, 'pytorch_model.bin')
                torch.save(self.model.state_dict(), path_model)
            else:
                model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                model_save.save_pretrained(path)
            print('Saving model: {}'.format(path))"""


            path = os.path.join(self.config.path_model_save_rank, f'step_{step_current}/')
            if not os.path.exists(path):
                os.makedirs(path)
            # 模型保存
            if self.config.model_name not in self.config.lm_model_list:
                path_model = os.path.join(path, 'pytorch_model.bin')
                torch.save(self.model.state_dict(), path_model)
            else:
                model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                model_save.save_pretrained(path)

            # 保存最优的模型
            if f1_eval > f1_best:
                # 创建文件夹
                path = os.path.join(self.config.path_model_save, 'step_best/')
                if not os.path.exists(path):
                    os.makedirs(path)
                # 模型保存
                if self.config.model_name not in self.config.lm_model_list:
                    path_model = os.path.join(path, 'pytorch_model.bin')
                    torch.save(self.model.state_dict(), path_model)
                # elif self.config.model_name == 'Erlangshen':
                #     new_state_dict = {}
                #     for key, value in self.model.state_dict.items():
                #         if 'Erlangshen.' in key:
                #             new_key = key.replace('Erlangshen.', '')
                #             new_state_dict[new_key] = value
                #         else:
                #             new_state_dict[key] = value
                #     torch.save(new_state_dict, path)
                else:
                    model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                    model_save.save_pretrained(path)
                f1_best = f1_eval
                print('Saving best model: {}\n'.format(path))
            # else:
            #     path = os.path.join(self.config.path_model_save, f'step_{step_current}/')
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     # 模型保存
            #     if self.config.model_name not in self.config.lm_model_list:
            #         path_model = os.path.join(path, 'pytorch_model.bin')
            #         torch.save(self.model.state_dict(), path_model)
            #     else:
            #         model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
            #         model_save.save_pretrained(path)

        return f1_best
    def get_embedding(self,text):
        text_enc=self.tokenizer.tokenizer(text,max_length=512, padding='max_length', return_tensors="pt", truncation="only_first")
        print(text_enc)

        input_ids=text_enc['input_ids'].to(self.device)
        attention_mask=text_enc['attention_mask'].to(self.device)
        # input_ids =torch.LongTensor(text_enc['input_ids']).to(self.device).unsqueeze(0)
        # attention_mask = torch.LongTensor(text_enc['attention_mask']).to(self.device).unsqueeze(0)
        output = self.model(input_ids, attention_mask)[1]
        embedding=output.tolist()
        print(embedding)
    def aggregate_results(self, step_current):
        world_size=int(os.environ["WORLD_SIZE"])
        """Rank 0 读取所有结果并计算 Recall@P"""
        all_dirs = [os.path.join(self.config.path_model_save, str(i), f'result_{step_current}.csv') 
                    for i in range(world_size)]
        
        # 仅保留存在的文件
        all_files = [f for f in all_dirs if os.path.exists(f)]
        if not all_files:
            print("No results found for aggregation.")
            return
        
        # 读取所有进程保存的 CSV 文件
        all_dfs = [pd.read_csv(f) for f in all_files]
        df = pd.concat(all_dfs, ignore_index=True)
        self.get_pr_multi(df,step_current)
    def get_pr_multi(self,df,step):
        precision_thresholds = [0.9, 0.8, 0.7, 0.6,0.5,0.4,0.3,0.2,0.1]

        end=0

        for id in range(len(self.config.multi_num_classes)):
            num_classes=self.config.multi_num_classes[id]
            new_df=df[df[f"label_{id}"] != -1]
            # print("eval id",id)
            for i in range(num_classes):
                # print("eval i",i)
                labels = np.array(new_df[f"label_{id}"]==i)  # 将条件判断结果转换为 NumPy 数组，布尔类型数组
                scores= np.array(new_df[f"score_{end+i}"]==i) 
                recall_at_precision=get_pr_ori(scores,labels)
                log_path = os.path.join(self.config.path_model_save, "eval.log")

                with open(log_path, "a") as log_file:  # 追加模式   
                    for p, v in recall_at_precision.items():
                        log_file.write(f"class_{id} label_{i} {p} {v}\n")         
            end+=num_classes
        

    def get_pr(self,df,step):
        precision_thresholds = [0.9, 0.8, 0.7, 0.6,0.5,0.4,0.3,0.2,0.1]
        recall_results = {}

        for i in range(self.config.num_classes):
            if f"score_{i}" not in df.columns:
                print(f"Warning: score_{i} not found in data, skipping.")
                continue

            scores = np.array(df[f"score_{i}"])  # 将列转换为 NumPy 数组
            labels = np.array(df["label"] == i)  # 将条件判断结果转换为 NumPy 数组，布尔类型数组

            # 按分数降序排列
            sorted_indices = np.argsort(scores)[::-1]
            sorted_labels = labels[sorted_indices]  # 使用排序的索引
            sorted_scores = scores[sorted_indices]  # 使用排序的索引

            recall_at_precision = {}
            total_selected = 0
            total_correct = 0
            score_thresh_map = {p: None for p in precision_thresholds}

            for j in range(len(sorted_scores)):
                total_selected += 1
                total_correct += sorted_labels[j]

                precision = total_correct / total_selected

                # 记录 Precision ≥ p 的最小分数阈值
                for p in precision_thresholds:
                    if precision >= p :
                        score_thresh_map[p] = sorted_scores[j]

            # 计算 Recall 只用一次遍历
            for p in precision_thresholds:
                score_thresh = score_thresh_map[p]
                if score_thresh is not None:
                    selected_labels = sorted_labels[sorted_scores >= score_thresh]
                    recall = np.sum(selected_labels) / np.sum(labels)
                else:
                    recall = 0.0  # 没有满足 Precision 的阈值

                recall_at_precision[f"class_{i}_P@{int(p*100)}"] = {"recall": recall,"score_thresh": score_thresh}

            recall_results[f"class_{i}"] = recall_at_precision


        log_path = os.path.join(self.config.path_model_save, "eval.log")

        with open(log_path, "a") as log_file:  # 追加模式
            log_file.write(f"\n=== Step {step} Precision-Recall Results ===\n")
            for cls, values in recall_results.items():
                log_file.write(f"{cls} Recall at Precision:\n")
                for p, v in values.items():
                    log_file.write(f"  {p}: Score Threshold = {v['score_thresh']}, Recall = {v['recall']}\n")
            log_file.write("=" * 50 + "\n")

        print(f"Results appended to {log_path}")


    def evaluate(self, data, print_table=False,step=0):
        """
        模型测试集效果评估
        """
        print("evaluate...")

        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        predict_score_all = np.array([], dtype=int).reshape(0,self.config.num_classes) 
        label_all = []
        note_id_all=[]
        text_all=[]
        taxonomy1_all=[]
        taxonomy2_all=[]
        
        if self.config.multi_num_classes != []:
            predict_score_all = np.array([], dtype=int).reshape(0,sum(self.config.multi_num_classes))
            label_all = np.array([], dtype=int).reshape(0,len(self.config.multi_num_classes))
            
        # loss_manager = LossManager(loss_type=self.config.loss_type, cl_option=False)
        # if self.config.distributed:#无法使用 
        #                 # 每个进程将处理的数据量
        #     rank = int(os.environ["RANK"])  # 当前进程编号            
        #     # 获取每个进程处理的数据样本数量
        #     num_samples = len(data.sampler.indices)  # sampler.indices 存储了当前进程处理的所有数据的索引
        #     print(f"Rank {rank} is processing {num_samples} samples")

        # 模型输入&输出
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data),total=len(data)):
                # if i == 10:
                #     break
                if self.config.slide_window:
                    split_content_list = [self.text_splitter(c) for c in batch['text']]
                    num_split_content_list = [len(x) for x in split_content_list]
                    split_content_list = list(itertools.chain.from_iterable(split_content_list))
                    
                    if self.config.add_cate:
                        text = data.dataset._text_to_encoding(split_content_list,repeat_and_concat([a + b for a, b in zip(batch['cate'], batch['cate2'])],num_split_content_list))
                    else:
                        text = data.dataset._text_to_encoding(split_content_list,cate=None)
                    input_ids = torch.LongTensor(text['input_ids']).to(self.device)
                    attention_mask = text['attention_mask'].to(self.device)
                else:
                    input_ids = batch['text_enc']['input_ids'].to(self.device)
                    attention_mask = batch['text_enc']['attention_mask'].to(self.device)
                output = self.model(input_ids, attention_mask)[0]
                # 计算loss
                # loss = F.cross_entropy(outputs, labels)
                # loss_total += outputx[0]
                if self.config.slide_window:
                    if self.config.multi_num_classes != []:
                        output_list = []
                        end = 0
                        for i,num_classes in enumerate(self.config.multi_num_classes):
                            output0 = output[i]  # 当前子任务的 logits
                            output0 = output0.split(num_split_content_list)  # 按组切
                            # 对每组取 max-pooling（dim=0 表示在 chunk 维度上）
                            output0 = torch.stack([x.max(dim=0)[0] for x in output0], dim=0)
                            output_list.append(output0)
                            end += num_classes

                        output = torch.cat(output_list, dim=1)
                        # output1, output2 = output[:, :2], output[:, 2:] 
                        # output1, output2 = output1.split(num_split_content_list), output2.split(num_split_content_list)
                        # output1, output2 = torch.stack([x.max(dim=0)[0] for x in output1], dim=0), torch.stack([x.max(dim=0)[0] for x in output2], dim=0)
                    else:
                        output = output.split(num_split_content_list)
                        output = torch.stack([x.max(dim=0)[0] for x in output], dim=0)
                # if self.config.slide_window:
                #     output = output.split(num_split_content_list)
                #     output = torch.stack([x.max(dim=0)[0] for x in output], dim=0)


                # target = batch['label'].to(self.device)
                # target1 = batch['label1'].to(self.device)
                # loss = loss_manager.compute(output, target)
                # loss_total += loss
                # 获取标签
                if self.config.multi_num_classes != []:
                    predict_score_list=[]
                    for i,num_classes in enumerate(self.config.multi_num_classes):
                        output=output_list[i]
                        predict_score = torch.nn.functional.softmax(output, dim=-1).cpu().numpy()  # 先 Softmax 再转 NumPy
                        predict_score_list.append(predict_score)
                    predict_score=np.concatenate(predict_score_list, axis=1)



                    # 计算预测类别
                    # predic = np.argmax(predict_score, axis=-1)  # Softmax 后选择最大概率的类别

                    predict_score_all=np.vstack([predict_score_all, predict_score])
                    # predict_all = np.append(predict_all, predic)
                    note_id_all.extend(batch['note_id'])
                    # taxonomy1_all.extend(batch['taxonomy1'])
                    
                    # text_all.extend(batch['text'])
                    label_all=np.vstack([label_all, batch['label']])
                    # taxonomy2_all.extend(batch['taxonomy2'])
                    # label = batch['label'].cpu().numpy()
                    # label1 = batch['label1'].cpu().numpy()
                    # label2 = batch['label2'].cpu().numpy()
                    # label3 = batch['label3'].cpu().numpy()

                    # predic = torch.max(output0, -1)[1].cpu().numpy()
                    # predic1 = torch.max(output1, -1)[1].cpu().numpy()
                    # predic2 = torch.max(output2, -1)[1].cpu().numpy()
                    # predic3 = torch.max(output3, -1)[1].cpu().numpy()

                    # labels_all, predict_all = np.append(labels_all, label), np.append(predict_all, predic)
                    # labels_all1, predict_all1 = np.append(labels_all1, label1), np.append(predict_all1, predic1)
                    # labels_all2, predict_all2 = np.append(labels_all2, label2), np.append(predict_all2, predic2)
                    # labels_all3, predict_all3 = np.append(labels_all3, label3), np.append(predict_all3, predic3)
                    
                    # label = batch['label'].cpu().numpy()
                    # predic1, predic2 = torch.max(output1, -1)[1].cpu().numpy()[label!=2], torch.max(output2, -1)[1].cpu().numpy()
                    # labels1 = label[label!=2]
                    # label[label==2] = 1
                    # labels2 = label
                    
                    # labels_all = np.append(labels_all, labels1)
                    # predict_all = np.append(predict_all, predic1)
                    # labels_all1 = np.append(labels_all1, labels2)
                    # predict_all1 = np.append(predict_all1, predic2)
                else:
                    predict_score = torch.nn.functional.softmax(output, dim=-1).cpu().numpy()  # 先 Softmax 再转 NumPy

                    # 计算预测类别
                    predic = np.argmax(predict_score, axis=-1)  # Softmax 后选择最大概率的类别

                    predict_score_all=np.vstack([predict_score_all, predict_score])
                    predict_all = np.append(predict_all, predic)
                    note_id_all.extend(batch['note_id'])
                    taxonomy1_all.extend(batch['taxonomy1'])
                    # taxonomy2_all.extend(batch['taxonomy2'])

                    text_all.extend(batch['text'])
                    label_all.extend(batch['label'])

                df = pd.DataFrame()

        # 添加 label 和 prediction 列
        df['note_id']=note_id_all
        # df["label"] = [ label.item() for label in label_all]
        # df["prediction"] = predict_all
        # df['text']=text_all
        # df['taxonomy1']=taxonomy1_all
        # df['taxonomy2']=taxonomy2_all

        # 处理 predict_score_all（将其拆分成 score_0, score_1, ...）
        if self.config.multi_num_classes != []:
            for i in range(len(self.config.multi_num_classes)):
                df[f"label_{i}"] = label_all[:, i] 
            for i in range(sum(self.config.multi_num_classes)):
                df[f"score_{i}"] = predict_score_all[:, i] 
        else:      
            df["label"] = [ label.item() for label in label_all]
            for i in range(self.config.num_classes):
                df[f"score_{i}"] = predict_score_all[:, i]

        # save_path=self.config.path_output_file
        # # 保存为 CSV
        # df.to_csv(save_path,index=False)
        # print(f"result to {save_path }") 

        if self.config.path_output_file=='':
            save_path = os.path.join(self.config.path_model_save_rank, f"result_{step}.csv")
        else:
            save_path=self.config.path_output_file
        df.to_csv(save_path, index=False)
        print(f"Result saved to {save_path}")

        if dist.is_initialized():
            dist.barrier()

        # Rank 0 处理最终结果
        if self.config.distributed and self.local_rank == 0:
            self.aggregate_results(step)      
        if dist.is_initialized():
            dist.barrier()
        
        # world_size = int(os.environ["WORLD_SIZE"])
        # note_id_all = torch.tensor(note_id_all) if not isinstance(note_id_all, torch.Tensor) else note_id_all
        # label_all = torch.tensor(label_all) if not isinstance(label_all, torch.Tensor) else label_all
        # predict_all = torch.tensor(predict_all) if not isinstance(predict_all, torch.Tensor) else predict_all
        # predict_score_all = torch.tensor(predict_score_all) if not isinstance(predict_score_all, torch.Tensor) else predict_score_all

        # # 将每个进程的结果收集到 rank 0 进程
        # gathered_note_id = [torch.zeros_like(note_id_all) for _ in range(world_size)]
        # gathered_label = [torch.zeros_like(label_all) for _ in range(world_size)]
        # gathered_predict = [torch.zeros_like(predict_all) for _ in range(world_size)]
        # gathered_predict_score = [torch.zeros_like(predict_score_all) for _ in range(world_size)]

        # # 将每个进程的数据收集到对应的 buffer 中
        # dist.all_gather(gathered_note_id, note_id_all)
        # dist.all_gather(gathered_label, label_all)
        # dist.all_gather(gathered_predict, predict_all)
        # dist.all_gather(gathered_predict_score, predict_score_all)

        # # 这里 rank 0 进程合并所有的数据
        # if rank == 0:
        #     note_id_all = torch.cat(gathered_note_id, dim=0).cpu().numpy()
        #     label_all = torch.cat(gathered_label, dim=0).cpu().numpy()
        #     predict_all = torch.cat(gathered_predict, dim=0).cpu().numpy()
        #     predict_score_all = torch.cat(gathered_predict_score, dim=0).cpu().numpy()

        #     # 创建 DataFrame
        #     df = pd.DataFrame()
        #     df['note_id'] = note_id_all
        #     df["label"] = label_all
        #     df["prediction"] = predict_all

        #     # 处理 predict_score_all（将其拆分成 score_0, score_1, ...）
        #     for i in range(predict_score_all.shape[1]):
        #         df[f"score_{i}"] = predict_score_all[:, i]

        #     # 保存为 CSV
        #     save_path = os.path.join(self.config.path_model_save, f"result_{step}.csv")
        #     df.to_csv(save_path, index=False)
        #     print(f"Result saved to {save_path}")
        return 1

        # 计算指标
        if self.config.multi_num_classes != []:
            f10 = metrics.f1_score(labels_all, predict_all, labels=[2], average=None)
            report = metrics.classification_report(labels_all, predict_all, target_names=['通过','正向','负向'], digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            print('\nEvaluate Classifier Performance '+'#'*50)
            logging.info(report)
            print(report)
            print('\nConfusion Matrix')
            logging.info(confusion)
            print(confusion)

            f11 = metrics.f1_score(labels_all1, predict_all1, labels=[1], average=None)
            report = metrics.classification_report(labels_all1, predict_all1, target_names=['通过','负向引战'], digits=4)
            confusion = metrics.confusion_matrix(labels_all1, predict_all1)
            print('\nEvaluate Classifier Performance '+'#'*50)
            logging.info(report)
            print(report)
            print('\nConfusion Matrix')
            logging.info(confusion)
            print(confusion)

            f12 = metrics.f1_score(labels_all2, predict_all2, labels=[1], average=None)
            report = metrics.classification_report(labels_all2, predict_all2, target_names=['通过','饭圈乱象'], digits=4)
            confusion = metrics.confusion_matrix(labels_all2, predict_all2)
            print('\nEvaluate Classifier Performance '+'#'*50)
            logging.info(report)
            print(report)
            print('\nConfusion Matrix')
            logging.info(confusion)
            print(confusion)

            f13 = metrics.f1_score(labels_all3, predict_all3, labels=[1], average=None)
            report = metrics.classification_report(labels_all3, predict_all3, target_names=['通过','极端对立'], digits=4)
            confusion = metrics.confusion_matrix(labels_all3, predict_all3)
            print('\nEvaluate Classifier Performance '+'#'*50)
            logging.info(report)
            print(report)
            print('\nConfusion Matrix')
            logging.info(confusion)
            print(confusion)
            print('#'*60)
            f1 = (f10 + f11 + f12 + f13) / 4
            # pdb.set_trace()
        else:
            acc = metrics.accuracy_score(labels_all, predict_all)
            f1 = metrics.f1_score(labels_all, predict_all, labels=[1], average='micro')
            # f11 = metrics.f1_score(labels_all, predict_all, labels=[2], average=None)
            # f12 = metrics.f1_score(labels_all, predict_all, labels=[1], average=None)
            # f1 = (f11 + f12) / 2
            print('\n>>Eval Set>>:  Acc:{}  MicroF1:{:.4f}'.format(acc, f1.item()))
            # {'micro', 'macro', 'samples','weighted', 'binary'}
            if print_table:
                # 打印指标
                report = metrics.classification_report(labels_all, predict_all, target_names=self.label, digits=4)
                confusion = metrics.confusion_matrix(labels_all, predict_all)
                print('\nEvaluate Classifier Performance '+'#'*50)
                logging.info(report)
                print(report)
                print('\nConfusion Matrix')
                logging.info(confusion)
                print(confusion)
                print('#'*60)
            
        return f1
    
    