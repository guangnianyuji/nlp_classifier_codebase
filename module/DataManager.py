
import re
import os
import random
import math
import numpy as np
import pandas as pd
import pickle as pkl
import pdb
import torch
# from tqdm.auto import tqdm
from datasets import Dataset, load_dataset 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
# from utils.IOOption import open_file, write_text, write_file
# from utils.textsplit import get_rank, get_world_size, init_distributed_mode
from module.ModelMap import map_tokenizer
from module.tokenizer.TextTokenizer import TextTokenizer
from module.tokenizer.LMTextTokenizer import LMTextTokenizer

from tqdm import tqdm
tqdm.pandas()  # 注册tqdm进度条支持
from util.pandas_util import repeat_data,round_data
class DataManager(object):
    
    def __init__(self, config):
        
        self.config = config
        self.tokenizer=config.tokenizer

        # self.load_label()               # 读取标签
        self.tokenizer=config.tokenizer # 读取tokenizer分词模型
    
    
    def get_dataset(self, file,data_type='train'):
        """
        获取数据集
        """
        # file = '{}.csv'.format(data_type)
        # file=data_type
        dataloader = self.data_process(file,data_type=data_type)
        return dataloader


    def data_process(self, file_name,data_type):
        # pdb.set_trace()
        if self.config.path_datasets is None:
            path=file_name
        else:
            path = os.path.join(self.config.path_datasets, file_name)
        if path.endswith('csv'):
            raw_datasets = pd.read_csv(path)
        elif os.path.isdir(path):
            # 是一个目录，读取其中所有 CSV 文件并拼接
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            dfs = []
            for f in csv_files:
                full_path = os.path.join(path, f)
                df = pd.read_csv(full_path)
                dfs.append(df)
            raw_datasets = pd.concat(dfs, ignore_index=True)
        print(f"load dataset from {path}, with {raw_datasets.shape[0]} examples")



        # raw_datasets = raw_datasets.fillna('')
        #raw_datasets = raw_datasets.drop_duplicates(subset='note_id', keep='first')
        # if not 'label' in raw_datasets.keys():
        #     raw_datasets['label']=0
        # if not 'note_id' in raw_datasets.keys():
        #     raw_datasets['note_id']=0        
        # print(raw_datasets['label'].value_counts().to_dict())
        # # raw_datasets['taxonomy1'].fillna('')
        # # raw_datasets['taxonomy2'].fillna('')
        # def check_type(s):
        #     if isinstance(s, float) or pd.isna(s):
        #         s = ''
        #     return s
        # print("check title")
        # raw_datasets['title']=raw_datasets['title'].progress_apply(check_type)
        # print("check content")
        # raw_datasets['content']=raw_datasets['content'].progress_apply(check_type)
        # print("check ocr")
        # raw_datasets['ocr']=raw_datasets['ocr'].progress_apply(check_type)

        # #不报错必须规定一下类型！！！
        # raw_datasets = raw_datasets[raw_datasets['note_id'].apply(lambda x: type(x)==str)]
        # def ab2(df):  
        #     title = df['title']
        #     content =  df['content']
        #     title_content = title + '。' + content
    
        #     if len(df['ocr']) > 0:
        #         text = title_content + '。' + df['ocr']
        #     else:
        #         text = title_content + '。' 
        #     return text   
        # print("concat to text")
        # raw_datasets['text']=raw_datasets.progress_apply(ab2, axis=1)

        # raw_datasets['label']=raw_datasets['topic'].apply(lambda x:1 if x!='通过' else 0)


        # if data_type=='train':
        #     raw_datasets=repeat_data(raw_datasets)
        #     raw_datasets=round_data(raw_datasets,self.config.world_size*self.config.batch_size)


        #     # 打乱数据集顺序
        #     raw_datasets = raw_datasets.sample(frac=1, random_state=42).reset_index(drop=True)

        # print("final labels.....")
        # print(raw_datasets['label'].value_counts().to_dict())

        # raw_datasets['taxonomy1']=raw_datasets['taxonomy1'].astype(str)
        # raw_datasets['taxonomy2']=raw_datasets['taxonomy2'].astype(str)  

        # raw_datasets = raw_datasets[raw_datasets['text'].apply(lambda x: len(x) <= 2000)]

        # if len(raw_datasets) > 400000:
        #     print('split...')
        #     chunksize = 10000  # 每次处理1万行数据
        #     chunks = pd.read_csv(path, chunksize=chunksize)

        #     text = np.array([])  # 创建一个空的numpy数组
        #     for chunk in tqdm.tqdm(chunks, total=400000 // chunksize):
        #         text_chunk = np.array(chunk['text'].tolist())
        #         text = np.concatenate([text, text_chunk])
        # else:


        #text = np.array(raw_datasets['text'].tolist())
        
        # raw_datasets['label'].fillna('通过', inplace=True)
        # raw_datasets['cate'].fillna('没有类目_假装是个类目', inplace=True)
        # raw_datasets['cate'] = raw_datasets['cate'].apply(lambda x: x[:15])
        # raw_datasets['cate'] = raw_datasets['cate'].astype('<U15')

        # #100条数据来for debug:
        # raw_datasets=raw_datasets.iloc[:100]
        # print(">>>>>>>>>>>>>>.iloc 100")
        
        #cate = np.array(raw_datasets['cate'].tolist())

        # if 'cate1' in raw_datasets.keys():
        #     cate1 = np.array(raw_datasets['cate1'].tolist())
        # else:
        #     cate1 = None
        
        # label = raw_datasets['label'].tolist()
        # note_id = raw_datasets['note_id'].tolist()
        #dataset = XHSDataset(self.tokenizer, raw_datasets, self.label2ids, self.config.max_length, self.config.multi_num_classes)
        dataset = fanquanDataset(tokenizer=self.tokenizer, pd_dataset=raw_datasets, label2ids=None, max_length=self.config.max_length, multi_num_classes=self.config.multi_num_classes)

        print("DISTRIBUTED??",self.config.distributed)
        if self.config.distributed:
            rank = int(os.environ["RANK"])  # 当前进程编号
            world_size = int(os.environ["WORLD_SIZE"])  # 总 GPU 数量
            local_rank = int(os.environ["LOCAL_RANK"])  # 当前 GPU 设备 ID
            print("world_size:",world_size, " rank:",rank)
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=8, sampler=sampler, shuffle=False, drop_last=False)
        else:
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        return dataloader
class fenwen1kw():
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        self.pd_dataset=pd_dataset 
        self.data_process()
        self.max_length = max_length
    def data_process(self):
        raw_datasets=self.pd_dataset
        if not 'note_id' in raw_datasets.keys():
            raw_datasets['note_id']="0"
        raw_datasets = raw_datasets[raw_datasets['note_id'].apply(lambda x: type(x)==str)]

        def parse_labels(row):
            labels = np.array([row['fanquan_label'], row['diyu_label'], row['gender_label'], row['negemo_label'], row['unkind_label']])
            return labels
        raw_datasets['label']=raw_datasets.progress_apply(lambda row:parse_labels(row),axis=1)
        raw_datasets=raw_datasets[raw_datasets['label'].apply(lambda x:np.sum(x)!=-5)]

        # raw_datasets=raw_datasets.iloc[:1000]
        print(raw_datasets.shape)

        self.pd_dataset=raw_datasets

    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def __getitem__(self, idx):
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}   

        item['idx'] = idx

        item['note_id'] = info_dict['note_id']

        item['text'] = info_dict['text1']
        item['label'] = info_dict['label'] 
        # item['taxonomy1']=info_dict['taxonomy1']
        # item['taxonomy2']=info_dict['taxonomy2']
        return item    
    def __len__(self):
        return self.pd_dataset.shape[0]  



        
        
class fanquanDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        self.pd_dataset=pd_dataset 
        self.data_process()
        self.max_length = max_length
    def data_process(self):
        #['note_id', 'title','content','ocr','first_image_ocr', 'taxonomy1','taxonomy2','gender','nickname','description','fans_num','author_taxonomy']
        raw_datasets=self.pd_dataset
        # raw_datasets=raw_datasets.iloc[:10]
        # if not 'label' in raw_datasets.keys():
        #     raw_datasets['label']=0
        if not 'note_id' in raw_datasets.keys():
            raw_datasets['note_id']=0        
        if not 'remarks' in raw_datasets.keys():
            raw_datasets['remarks']=''
        # print(raw_datasets['label'].value_counts().to_dict())
        raw_datasets['taxonomy1']=raw_datasets['taxonomy1'].fillna('')
        raw_datasets['taxonomy2']=raw_datasets['taxonomy2'].fillna('')
        def check_type(s):
            if isinstance(s, float) or pd.isna(s):
                s = ''
            return s
        print("check title")
        raw_datasets['title']=raw_datasets['title'].progress_apply(check_type)
        print("check content")
        raw_datasets['content']=raw_datasets['content'].progress_apply(check_type)
        print("check ocr")
        raw_datasets['ocr']=raw_datasets['ocr'].progress_apply(check_type)
        print("check first ocr")
        raw_datasets['first_image_ocr']=raw_datasets['first_image_ocr'].progress_apply(check_type)

        #不报错必须规定一下类型！！！
        raw_datasets = raw_datasets[raw_datasets['note_id'].apply(lambda x: type(x)==str)]
        def ab2(df):  
            text=f"类目：{df['taxonomy1']},{df['taxonomy2']}\n标题：{df['title']}\n正文：{df['content']}\n封面图文字：{df['first_image_ocr']}\n所有图片文字：{df['ocr']}"
            return text
        print("concat to text")
        raw_datasets['text']=raw_datasets.progress_apply(ab2, axis=1)

        label_map={"通过":0,"负向引战":1,"饭圈乱象":2,"正向声援":0,"体育饭圈磕cp":3,"极端对立":4}

        raw_datasets['label'] = raw_datasets['remarks'].apply(lambda x: label_map[x] if x in label_map else 0)
        raw_datasets[['note_id', 'text', 'taxonomy1']] = raw_datasets[['note_id', 'text', 'taxonomy1']].astype(str)
        raw_datasets['label'] = raw_datasets['label'].astype(int)

        # 打印列的数据类型以验证
        print(raw_datasets.dtypes)
        print("final labels.....")
        print(raw_datasets['label'].value_counts().to_dict())
        self.pd_dataset=raw_datasets


    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def __getitem__(self, idx):
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}   
        item['idx'] = idx

        item['note_id'] = info_dict['note_id']

        item['text'] = info_dict['text']
        item['label'] = info_dict['label'] 
        item['taxonomy1']=info_dict['taxonomy1']
        # item['taxonomy2']=info_dict['taxonomy2']
        return item    
    def __len__(self):
        return self.pd_dataset.shape[0]  
class isgenderDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        self.pd_dataset=pd_dataset 
        self.max_length = max_length
    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def __getitem__(self, idx):
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}   
        item['idx'] = idx

        item['note_id'] = info_dict['note_id']

        item['text'] = info_dict['text']
        item['label'] = info_dict['label'] 
        # item['taxonomy1']=info_dict['taxonomy1']
        # item['taxonomy2']=info_dict['taxonomy2']
        return item    
    def __len__(self):
        return self.pd_dataset.shape[0]
class enAttackDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        self.pd_dataset=pd_dataset 
        self.max_length = max_length
    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def __getitem__(self, idx):
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}   
        item['idx'] = idx

        item['note_id'] = info_dict['note_id']

        text = info_dict['text']
        item['text_save'] = text #not pass tokenzier, for sliding_window

        #  text = re.sub(r'\b[bB][iI][tT][cC][hH]\b', '', text, flags=re.IGNORECASE)
        item['text'] = text

        text_enc = self._text_to_encoding(text,'')
        item_text = {key: val.clone().detach() for key, val in text_enc.items()}
        item['text_enc'] = item_text   

        item['label'] = info_dict['label'] 
        # item['taxonomy1']=info_dict['taxonomy1']
        # item['taxonomy2']=info_dict['taxonomy2']
        return item        

    def __len__(self):
        return self.pd_dataset.shape[0]
        
class ttChineseDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        self.pd_dataset=pd_dataset
        self.label2ids = label2ids
        # self.note_id = note_id
        self.max_length = max_length
    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def __getitem__(self, idx):
        """
        add_column = ['note_id', 'text', 'cate', 'label', 'image_url_list', 'note_type']
        """
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}

        item['idx'] = idx
        item['note_id'] = info_dict['note_id']

        content = info_dict['text']
        item['content'] = content #not pass tokenzier, for sliding_window

        text_enc = self._text_to_encoding(content,'')
        item_text = {key: val.clone().detach() for key, val in text_enc.items()}
        item['text'] = item_text       


        cate = info_dict['cate']
        item['cate']=cate

        item['label'] = str(info_dict['label'])

        item['taxonomy1']=info_dict['taxonomy1']
        item['taxonomy2']=info_dict['taxonomy2']
        return item


    def __len__(self):
        return self.pd_dataset.shape[0]


class XHSDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pd_dataset, label2ids,max_length, multi_num_classes):
        self.tokenizer = tokenizer
        # self.text = text
        # self.cate = cate
        # self.cate1 = cate1
        # self.label = label
        self.pd_dataset=pd_dataset
        self.label2ids = label2ids
        # self.note_id = note_id
        self.max_length = max_length
        self.multi_num_classes = multi_num_classes

    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    
    def __getitem__(self, idx):
        """
        add_column = ['note_id', 'text', 'cate', 'label', 'image_url_list', 'note_type']
        """
        info_dict = dict(self.pd_dataset.iloc[idx])
        item = {}
        content = info_dict['text']
        cate = info_dict['cate']
        label = str(info_dict['label'])
        item['label_str'] = label
        if 'cate1' in self.pd_dataset.keys():
            cate1 =  info_dict['cate1']
            item['cate1'] = cate1
            """
            if cate1 not in ['无', '不命中']:
                cate1 = '命中'
            """
            text_enc = self._text_to_encoding(content, cate1)
            if label not in self.label2ids.keys():
                label = '0'
        else:
            #text_enc = self._text_to_encoding(content, cate)
            text_enc = self._text_to_encoding(content,'')
            if self.multi_num_classes != []:
                label_dict = {
                    '通过': [0, 0, 0, 0],
                    '正向声援': [1, 0, 0, 0],
                    '负向引战': [2, 1, 0, 0],
                    '饭圈乱象': [2, 0, 1, 0],
                    '极端对立': [2, 0, 0, 1]
                }
                label, label1, label2, label3 = label_dict[item['label_str']]
            else:
                pass
                # if label in '负向引战+极端对立+饭圈乱象':
                #     label = '负向引战+极端对立+饭圈乱象'
      
        
        #item_text = {key: torch.tensor(val) for key, val in text_enc.items()}
        item_text = {key: val.clone().detach() for key, val in text_enc.items()}
        item['text'] = item_text
        item['content'] = content
        if len(cate.split("_"))>0:
            item['cate'] = cate.split("_")[0]
        else:
            item['cate'] = ''
        if len(cate.split("_"))>1:
            item['cate2'] = cate.split("_")[1]
        else:
            item['cate2'] = ''      
        item['note_id'] = info_dict['note_id']
        # try:
        #     item['label'] = torch.tensor(self.label2ids[label])
        # except:
        #     pdb.set_trace()
        if self.multi_num_classes != []:
            item['label'] = torch.tensor(label)
            item['label1'] = torch.tensor(label1)
            item['label2'] = torch.tensor(label2)
            item['label3'] = torch.tensor(label3)
        else:
            item['label'] = torch.tensor(self.label2ids[label])
        #item['label'] = torch.tensor(self.label[idx])
        #item['label'] = torch.LongTensor(self.label[idx])
        item['idx'] = idx
        return item

    def __len__(self):
        return self.pd_dataset.shape[0]