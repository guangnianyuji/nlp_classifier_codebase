import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from tqdm import tqdm
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, BertConfig, AutoConfig
import onnx
from onnxconverter_common import float16
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
#!pip install transformers==4.9.2

class Config():
    slide_window=True
    add_cate=False
    model_path="/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/logs/train/SunyifeiTanghuang48LConfig/20250321_153747/checkpoints/0/step_55000"
    initial_pretrain_path="/mnt/public03/usr/sunyifei2/level_classification_newstd/checkpoints/chinese-bert-wwm-ext"
    num_classes=2

class oriPredictor:
    def __init__(self,conifg):
        self.config=config

        self.tokenizer=load_tokenizer()
        self.model_path=self.config.model_path
        self.initial_pretrain_path=self.config.initial_pretrain_path

        load_model()

        self.text_splitter=TextSplitter()

    
    def load_model(self):
        self.model=Bert
        self.model = self.model.from_pretrained(self.initial_pretrain_path)
        if not self.model.fc[-1].out_features == self.config.num_classes:
            print('The classifier layer reinitialized')
            self.model.fc[-1] = torch.nn.Linear(in_features=self.model.fc[-1].in_features, out_features=self.config.num_classes, bias=True)  
        path_model=os.path.join(self.model_path,"pytorch_model.bin")

        if os.path.isfile(path_model):
            msg=self.model.load_state_dict(torch.load(path_model))
            print(f">>>>>>>>>>>>>>>>>>load weight from {path_model}, with msg :{msg}") 
        else:
            raise Exception
    def _text_to_encoding(self, content, cate):
        # return self.tokenizer.tokenizer(content, max_lenth=400)
        return self.tokenizer.tokenizer(content, cate, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation="only_first")
    def predict_one_data(self,data):
        note_info=data
        output_dict, fea_status = fea_process_fuc(note_info,self.tokenizer)  
        input_ids=torch.tensor(output_dict['ids'])
        attention_mask=torch.tensor(output_dict['att'])
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
        output=output[0]
        print(">>>>>>>>output")
        print(output)
        if self.config.slide_window:
            output=output.max(dim=0)[0]
            print(output)
        output = torch.nn.functional.softmax(output,dim=-1).cpu().detach().numpy()
        print(output)



        
class Predictor:
    def __init__(self,onnx_fp16_path):
        self.tokenizer=load_tokenizer()
        self.onnx_fp16_path=onnx_fp16_path
        self.load_sess()
    def load_sess(self):
        self.sess=InferenceSession(self.onnx_fp16_path,providers=['CUDAExecutionProvider'])
    def predict_one_data(self,data):
        note_info=data
        output_dict, fea_status = fea_process_fuc(note_info,self.tokenizer)
        #logger.info(str(output_dict))
        if output_dict is not None and output_dict != {}:
            input_data = {
                "INPUT_0":output_dict['ids'],
                "INPUT_1":output_dict['att']
            }
            taxonomy1 = output_dict['taxonomy1']
            taxonomy2 = output_dict['taxonomy2']
            prob_list = self.sess.run(output_names=["OUTPUT_0"],input_feed=input_data)
            prob_list=np.array(prob_list[0])
            # print(prob_list.shape)#(1, 1, 2)
            # prob_list=model_output['OUTPUT_0']
            score=self.gender_post_process(prob_list)
        else:
            pass       
        return score
    def predict_list(self,data_list,save_path):
        note_id_list=[]
        score_list=[]
        for data in tqdm(data_list):
            score=self.predict_data(data)
            note_id_list.append(data['note_id'])
            score_list.append(score)
        df=pd.DataFrame({'note_id':note_id_list,'score':score_list})
        df.to_csv(save_path,index=False)
    
    def gender_post_process(self, prob_list):
        prob_list = torch.tensor(prob_list)
        print(">>>>>>prob list")
        print(prob_list)
        if prob_list.shape[0] == 1:
            norm_prob_list = torch.nn.functional.softmax(prob_list,dim=1)
            norm_prob_list = norm_prob_list.tolist()
            #print(norm_prob_list)
            print('result:',norm_prob_list)
        else:
            prob_list = prob_list.max(dim=0)[0].unsqueeze(dim=0)
            norm_prob_list = torch.nn.functional.softmax(prob_list)
            norm_prob_list = norm_prob_list.tolist()
            print('result:',norm_prob_list)
        score = norm_prob_list[0][1]
        return score


def convert_fp16(onnx_model_path,onnx_model_fp16_path):
    new_onnx_model = convert_float_to_float16_model_path(onnx_model_path, keep_io_types=True)

    # Convert to FP16

    onnx.save(new_onnx_model, onnx_model_fp16_path, save_as_external_data=True)

    print(f"Converted ONNX model to FP16: {onnx_model_fp16_path}")

def test_fp_16(onnx_model_fp16_path,tokenizer,device="cuda"):
    content_list=[
        "inputs = tokenizer(content_list, return_tensors=, truncation=True, max_length=256, padding='max_length')inputs = tokenizer(content_list, return_tenso"", truncation=True, max_length=256, padding='max_length')inputs = tokenizer(content_list, return_tensors truncation=True, max_length=256, padding='max_length')"
    ]
    text_enc = tokenizer(content_list, return_tensors="pt", truncation=True, max_length=256, padding='max_length')
    encode_dict = {key: val.cpu().numpy() for key, val in text_enc.items()}

    input_ids = encode_dict['input_ids']
    input_masks = encode_dict['attention_mask']
    input_data = {
        "INPUT_0":input_ids,
        "INPUT_1":input_masks
    }    
    print(input_data)
    sess=InferenceSession(onnx_model_fp16_path,providers=['CUDAExecutionProvider'])
    prob_list = sess.run(output_names=["OUTPUT_0"],input_feed=input_data)
    prob_list=np.array(prob_list[0])
    print(prob_list)


def convert_onnx(model,tokenizer,onnx_path,device="cuda"):
    content_list=[
        "inputs = tokenizer(content_list, return_tensors=, truncation=True, max_length=256, padding='max_length')inputs = tokenizer(content_list, return_tenso"", truncation=True, max_length=256, padding='max_length')inputs = tokenizer(content_list, return_tensors truncation=True, max_length=256, padding='max_length')"
    ]
    inputs = tokenizer(content_list, return_tensors="pt", truncation=True, max_length=256, padding='max_length')
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_masks = torch.LongTensor(inputs['attention_mask']).to(device)
    model.to(device)
    dummpy_input = {
    "INPUT_0":  input_ids,
    "INPUT_1": attention_masks,
    }   
    torch.onnx.export(model, 
        tuple(dummpy_input.values()), 
        onnx_path, 
        export_params=True,
        opset_version=14,
        input_names=list(dummpy_input), 
        output_names = ['OUTPUT_0'],
        dynamic_axes= {'INPUT_0': {0: 'batch_size'},
                    'INPUT_1': {0: 'batch_size'},
                    'OUTPUT_0': {0: 'batch_size'}})
    
    print(f'>>>>>>>>>>>Model has been converted to ONNX {onnx_path}' )   


from module.ModelMap import map_model,map_tokenizer
from module.Trainer import load_state
def load_model():
    #model_name='Bert'large
    model_name='MegatronBert_t'
    initial_pretrain_model='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'  
    initial_pretrain_tokenizer='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'  
    path_model='/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/logs/train/SunyifeiTanghuang48LConfig/20250321_153747/checkpoints/0/step_55000/model.safetensors.index.json'
    model=None
    tokenizer=None
    if model_name in ('MegatronBert_t','XLMRoberta'):
        model = map_model(model_name)
        model_config = AutoConfig.from_pretrained(initial_pretrain_model, num_labels=5)  
        model=model.from_pretrained(initial_pretrain_model,config=model_config)
        # num_classes=2
        # if not model.fc[-1].out_features== num_classes:
        #     print(f"change num_classes!!!! {num_classes}")
        #     model.fc = torch.nn.Linear(in_features= model.classifier.in_features, out_features=num_classes, bias=True) 
        msg=model.load_state_dict(load_state(path_model))
        print("load msg",msg)


        tokenizer = map_tokenizer(model_name)
        tokenizer = tokenizer.from_pretrained(initial_pretrain_tokenizer)
    else:
        tokenizer=BertTokenizer.from_pretrained(initial_pretrain_tokenizer)
    return model,tokenizer

if __name__ == '__main__':
    # model = AutoModelForSequenceClassification.from_pretrained("/mnt/public03/usr/sunyifei2/hugging_face/models--KoalaAI--Text-Moderation")
    model,tokenizer=load_model()
    # # # tokenizer = AutoTokenizer.from_pretrained('/mnt/public03/usr/sunyifei2/hugging_face/chinese-bert-wwm-ext')
    onnx_path='/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/logs/train/SunyifeiTanghuang48LConfig/20250321_153747/checkpoints/0/step_55000/onnx/model.onnx'
    onnx_fp16_path='/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/logs/train/SunyifeiTanghuang48LConfig/20250321_153747/checkpoints/0/step_55000/fp16/model_fp16.onnx'
    convert_onnx(model=model,tokenizer=tokenizer,onnx_path=onnx_path)
    convert_fp16(onnx_model_path=onnx_path,onnx_model_fp16_path=onnx_fp16_path)
    test_fp_16(onnx_model_fp16_path=onnx_fp16_path,tokenizer=tokenizer)
    # onnx_fp16_path="/mnt/public03/usr/sunyifei2/checkpoints/1205_youdao_video/step_22000/onnx_model/model_fp16.onnx"
    # predictor=Predictor(onnx_fp16_path=onnx_fp16_path)

    
    # test_file_path='/mnt/public03/usr/sunyifei2/engagement_bait/qs_deploy/check_datasets/1211_youdao_v1/test_for_online_1024_input_data_image_list.json'
    # with open(test_file_path, 'r',encoding='utf-8-sig') as f:
    #     datas = json.load(f)
    # save_path='/mnt/public03/usr/sunyifei2/engagement_bait/qs_deploy/check_datasets/1211_youdao_v1/ori_result_1212.csv'
    #predictor=Predictor(onnx_fp16_path=onnx_fp16_path)
    #predictor.predict(datas,save_path)
    # data= {
    #     "note_id": "67359a100000000019015804",
    #     "title": "组了一个全国禅修高质量创业交流qun",
    #     "content": "🎈一定要给大家分享一个超棒的禅修创业交流群🥳。\t\n✨在这里，前后端的禅修创业者汇聚一堂，有创始人👨‍💼、合伙人👩‍🤝‍👨、投资人💼、运营人等等。大家真诚交流，毫无保留地分享创业经验💡、实战干货📝和最新打法🎯。\t\n💪群里的氛围积极向上，大家一起链接资源🎗️，共享禅修的行业资源，寻找合作机遇🎉。在这里，你不再是一个人在战斗，而是有一群志同道合的伙伴与你并肩前行🚶‍♂️🚶‍♀️。\t\n🌟除了线上的热烈交流，我们还会不定期举办线下聚会交流🥳。让你有机会面对面交流，拓展人脉资源👥。\t\n🙌欢迎加入这样一个积极、长期、认真交流的禅修创业圈子\n #禅修[话题]#  #干货分享[话题]#  #旅修[话题]#  #创业三十六计[话题]#  #资源整合[话题]# \t\n",
    #     "taxonomy1": "职场",
    #     "taxonomy2": "职场行业",
    #     "ocr": ", 禅修旅修, 资源交换, 老地, 法",
    #     "note_type": "1",
    #     "asr": "",
    #     "rn_b": 1,
    #     "note_id_d": "67359a100000000019015804",
    #     "ocr_image_text": "{\"http://ci.xiaohongshu.com/spectrum/1040g34o31a618t1jmm0g5pnoptlnc1ugm120le8?imageView2/2/w/1080/format/jpg\":\", 禅修旅修, 资源交换, 老地, 法\"}",
    #     "video_first_image_url": "http://ci.xiaohongshu.com/spectrum/1040g34o31a618t1jmm0g5pnoptlnc1ugm120le8?imageView2/2/w/1080/format/jpg",
    #     "image_url_list": "http://ci.xiaohongshu.com/spectrum/1040g34o31a618t1jmm0g5pnoptlnc1ugm120le8?imageView2/2/w/1080/format/jpg",
    #     "rn_d": 1
    # }
    # data=    {
    #     "note_id": "673598f300000000190198c8",
    #     "title": "素龙这么贵 到底谁在买啊",
    #     "content": "标题党哦 想入一个素龙挂件 小翅膀的戴头套的在犹豫买哪个\n看演唱会前：一个挂件 有点贵 \n看演唱会后：素龙儿子好可爱啊 性价比超高哒",
    #     "taxonomy1": "时尚",
    #     "taxonomy2": "配饰",
    #     "ocr": ", 14:28, <, 搜本店商品, 搜索, 素龙营业中周边店ξ, 新店开业 已完成资质认证, +关注, 2.5万, 1188, 2个作品>, 粉丝, 销量, 抖音号, 综合, 销量, 新品, 价格, 万·伏特, RowanTic, 汪苏泷, RowanTic, Rowan Tic包子头套龙毛绒挂件ξ, 十万达特主题素龙挂件（闪电版）, 【官方正版】RowanTic包, *效果图仅供参考，以实物材质及大小为准, 热搜度超99%同类品, 【官方正版】RowanTic十, 热搜度超99%同类品, ￥98开售价759人收藏, ￥119开售价 439人收藏, 7天无理由退货 极速退款, 7天无理由退货 极速退款, 万·伏特, RowanTicξ, 汪苏泷, RowanTic素龙毛绒公仔, 十万特心电感应援棒（左）, 【官方正版】RowanTic素, 【官方正版】RowanTic十, 首页, 全部商品, 联系客服",
    #     "note_type": "1",
    #     "asr": "",
    #     "rn_b": 1,
    #     "note_id_d": "673598f300000000190198c8",
    #     "ocr_image_text": "{\"http://ci.xiaohongshu.com/1040g2sg31a6158hl72705nrhnk308pq0l2l9rio?imageView2/2/w/1080/format/jpg\":\", 14:28, <, 搜本店商品, 搜索, 素龙营业中周边店ξ, 新店开业 已完成资质认证, +关注, 2.5万, 1188, 2个作品>, 粉丝, 销量, 抖音号, 综合, 销量, 新品, 价格, 万·伏特, RowanTic, 汪苏泷, RowanTic, Rowan Tic包子头套龙毛绒挂件ξ, 十万达特主题素龙挂件（闪电版）, 【官方正版】RowanTic包, *效果图仅供参考，以实物材质及大小为准, 热搜度超99%同类品, 【官方正版】RowanTic十, 热搜度超99%同类品, ￥98开售价759人收藏, ￥119开售价 439人收藏, 7天无理由退货 极速退款, 7天无理由退货 极速退款, 万·伏特, RowanTicξ, 汪苏泷, RowanTic素龙毛绒公仔, 十万特心电感应援棒（左）, 【官方正版】RowanTic素, 【官方正版】RowanTic十, 首页, 全部商品, 联系客服\"}",
    #     "video_first_image_url": "http://ci.xiaohongshu.com/1040g2sg31a6158hl72705nrhnk308pq0l2l9rio?imageView2/2/w/1080/format/jpg",
    #     "image_url_list": "http://ci.xiaohongshu.com/1040g2sg31a6158hl72705nrhnk308pq0l2l9rio?imageView2/2/w/1080/format/jpg",
    #     "rn_d": 1
    # }


    ##ori
    config=Config()
    # predictor=oriPredictor(config)
    #
    # onnx_fp16_path="/mnt/public03/usr/sunyifei2/checkpoints/1205_youdao_video/step_22000/onnx_model/model_fp16.onnx"
    predictor=Predictor(onnx_fp16_path)
    data=     {
        "note_id": "678c45410000000019004204",
        "title": "",
        "content": "第一次发帖，有点紧张\n#fyp[话题]# #tiktok[话题]# #ban[话题]# #你好[话题]# #嗡嗡声[话题]# ",
        "note_type": "1",
        "ocr_image_text": "{\"http://ci.xiaohongshu.com/1040g00831cqlmark0c005psc80a393gc1j99mjg?imageView2/2/w/1080/format/jpg\":\", Ni hao huzz\"}",
        "video_first_image_url": "http://ci.xiaohongshu.com/1040g00831cqlmark0c005psc80a393gc1j99mjg?imageView2/2/w/1080/format/jpg",
        "image_url_list": "http://ci.xiaohongshu.com/1040g00831cqlmark0c005psc80a393gc1j99mjg?imageView2/2/w/1080/format/jpg",
        "taxonomy1": "生活记录",
        "taxonomy2": "接地气生活",
        "asr": ""
    }
    predictor.predict_one_data(data)


    
    

