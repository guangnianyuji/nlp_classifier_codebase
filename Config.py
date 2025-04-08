import os
import random
 
from module.models.Transformer import Transformer
 
 
class Config(object):
        
    # 运行模式
    mode = 'test'
    
    # GPU配置
    
    cuda_visible_devices = '0,1,2,3'                                  # 可见的GPU
    device = 'cuda:0'                                           # master GPU
    # port = str(random.randint(10000,60000))                     # 多卡训练进程间通讯端口
    init_method = 'env://'                     # 多卡训练的通讯地址
    world_size = 1                                       # 线程数，默认为1
    
    # 模型选型
    # 基础模型：FastText/TextCNN/TextRNN/TextRCNN/Transformer
    # 语言模型：Bert/Albert/Roberta/Distilbert/Electra/XLNet
    model_name='Bert'     
    # initial_pretrain_model = '/mnt/public/usr/ningweiyu/checkpoints/pretrain/tanghuang_48L'           
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/chinese-bert-wwm-ext'  
    # initial_pretrain_model = '/mnt/public/usr/ningweiyu/checkpoints/pretrain/MegatronBert24L-NegativeEmotionV2-mapping'  
    # initial_pretrain_model = 'bert-base-chinese'         
    # initial_pretrain_model = '/mnt/public/usr/ningweiyu/checkpoints/fanquan/tanghuang_focal_5fenlei_0929_bs16/step_best' 
 
    # initial_pretrain_tokenizer = '/mnt/public/usr/ningweiyu/checkpoints/pretrain/MegatronBert24L-NegativeEmotionV2-mapping'
    # initial_pretrain_tokenizer = '/mnt/public/usr/ningweiyu/checkpoints/pretrain/tanghuang_48L'
    # initial_pretrain_tokenizer = '/mnt/public/usr/ningweiyu/checkpoints/fanquan/tanghuang_focal_5fenlei_0929_bs16/step_best'
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/chinese-bert-wwm-ext'     
    #initial_pretrain_tokenizer = 'bert-base-chinese'       # 加载的预训练模型checkpoint
    resume_from = ''#'/mnt/public/usr/ningweiyu/checkpoints/pray/bert_5fenlei_0826_bs64_12L_data15w/step_best'
    lm_model_list = ['Bert','Albert','Roberta','Distilbert','Electra','XLNet', 'MegatronBert', 'Erlangshen', 'MegatronBert_t', 'Erlangshen_24L','XLMRoberta','MegatronBert_t_multi']
    
    # 训练配置
    num_epochs = 5                                             # 迭代次数
    batch_size = 256                                    # 每个批次的大小
    learning_rate = 1e-5                                        # 学习率
    num_warmup_steps = 0.1                                      # warm up步数
    max_length = 256                                         # 句子最长长度
    padding = True                                              # 是否对输入进行padding
    step_save = 500                                          # 多少步保存一次模型
    loss_type = 'focalloss'#focalloss' #'ce'#'FocalLossWithDenoise'
    add_cate = False
    n_vocab = 10000
    embed = 256
    dropout = 0.01
    hidden_size = 128
    num_classes = 2
    multi_num_classes = []#[3,3]
    in_features = 2048#768*2
    distributed = False
    
    # 对比学习
    cl_option = False                                           # 是否使用对比学习
    cl_method = 'Rdrop'                                         # Rdrop/InfoNCE
    cl_loss_weight = 0.5                                        # 对比学习loss比例
    # 对抗训练
    adv_option = 'None'                                         # 是否引入对抗训练：none/FGM/PGD
    adv_name = 'word_embeddings'
    adv_epsilon = 1.0
    # 混合精度训练
    fp16 = False
    fp16_opt_level = 'O1'                                   # 训练可选'O1'，测试可选'O3'
    #滑动窗口
    slide_window = True
    which_step = 'step_13000'
    convert_onnx = False
    
    class_file_path="configfile/class.txt"
    # 模型及路径配置
    path_root = os.getcwd()
    train_dataset = 'train_1108_novideo'
    val_dataset = 'val_1108_novideo'
    # test_dataset = 'test_0612_1d_10w_add1case'
    test_dataset = "/mnt/public03/usr/sunyifei2/engagement_bait_datasets/20241129/test_dataset.csv"
 
    path_model_save = '/mnt/public/usr/ningweiyu/checkpoints/inter/Bert_12L_64bs'
   
     # 模型保存路径   
    path_datasets = '/mnt/public/usr/ningweiyu/datasets/train/inter/'
    path_log = os.path.join(path_root, 'logs')
    path_output = '/mnt/public03/usr/sunyifei2/risk_text_classifier/logs/SunyifeiBertConfig/20241129_165131/test_output'
 
    path_output_file = os.path.join(path_output, f'{test_dataset}_{which_step}_res.csv')


    use_loss_weight=False
class SunyifeiUnfriend(Config):
    model_name='MegatronBert_t'     
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/tanghuang_48L'    
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/tanghuang_48L'    

class onnxConfig(Config):
    test_dataset = '/mnt/public03/usr/sunyifei2/checkpoints/youdaohudong/test_for_convert_onnx.csv'
    mode = 'onnx'
    path_model_save=None
    test_model_path="/mnt/public03/usr/sunyifei2/risk_text_classifier/logs/SunyifeiBertConfig/20241203_022308/checkpoints/step_22000"
    path_datasets = None
    batch_size =1

class SunyifeiBertConfig(Config):
    initial_pretrain_model='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/chinese-bert-wwm-ext'
    initial_pretrain_tokenizer='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/chinese-bert-wwm-ext'
    train_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/data_2.csv'
    val_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/val.csv'
    model_name='Bert'     
    batch_size = 32
    test_dataset=None  
    path_datasets = None    
    mode='train'
    distributed =True
    class_file_path='/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/class.txt'
    step_save= 5000
    num_classes=2
class SunyifeiTanghuang48LConfig(Config):
    initial_pretrain_model='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'
    initial_pretrain_tokenizer='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'
    model_name='MegatronBert_t'     
    batch_size = 4
    test_dataset=None  
    path_datasets = None    
    mode='train'
    distributed =True
    # class_file_path='/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/class.txt'
    step_save= 5000
    num_classes=5
class SunyifeiMultiTanghuang48LConfig(Config):
    initial_pretrain_model='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'
    initial_pretrain_tokenizer='/mnt/nj-larc/usr/ajie1/sunyifei/hugging_face/tanghuang_48L'
    model_name='MegatronBert_t_multi'     
    batch_size = 4
    test_dataset=None  
    path_datasets = None    
    mode='train'
    distributed =True
    # class_file_path='/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/class.txt'
    step_save= 10000
    multi_num_classes=[2,2,2,2,2]

class SunyifeiBertTTChinese(Config):
    train_dataset = None
    val_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/20250118data/val_0118.csv'
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/chinese-bert-wwm-ext'  
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/chinese-bert-wwm-ext'    
    distributed=False
    resume_from='/mnt/public03/usr/sunyifei2/checkpoints/20250118_tt_chinese/step_20000'
    class_file_path='/mnt/public03/usr/sunyifei2/datasets/20250116_tt_Chinese/class.txt'
class SunyifeiBertEnNegative(Config):
    train_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250120_en_negative/data.csv'
    val_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250120_en_negative/val.csv'
    test_dataset=None  
    
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/models--google-bert--bert-base-uncased'  
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/models--google-bert--bert-base-uncased'    

    path_datasets = None    
    distributed=True
    step_save=5000
    num_epochs = 1  

    mode='train'
    batch_size = 32
    world_size = 4  

class SunyifeiXLMRoberta(Config):
    model_name='XLMRoberta'

    train_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250122_hatebase/data_437420.csv'
    val_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250120_en_negative/val.csv'
    test_dataset=None  
    
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/models--FacebookAI--xlm-roberta-large'  
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/models--FacebookAI--xlm-roberta-large'    

    path_datasets = None    
    distributed=True
    step_save=1000
    num_epochs = 2

    mode='train'
    batch_size = 4
    num_classes = 2

    use_loss_weight=False

    loss_type='ce'
    # world_size = 4  
class SunyifeiXLMRobertaTest(Config):
    model_name='XLMRoberta'

    train_dataset = None
    val_dataset = '/mnt/public03/usr/sunyifei2/datasets/20250123_en_negative_40w/val.csv'
    test_dataset=None  
    
    initial_pretrain_model = '/mnt/public03/usr/sunyifei2/hugging_face/models--FacebookAI--xlm-roberta-large'  
    initial_pretrain_tokenizer = '/mnt/public03/usr/sunyifei2/hugging_face/models--FacebookAI--xlm-roberta-large'    

    path_datasets = None    
    distributed=False
    step_save=5000
    num_epochs = 10     

    mode='infer'
    batch_size = 64
    world_size = 4  

    resume_from="/mnt/public03/usr/sunyifei2/risk_text_classifier/logs/SunyifeiXLMRoberta/20250122_164928/checkpoints/step_10000"

class SunyifeiBertTestConfig(Config):
    train_dataset = None
    val_dataset = None
    test_dataset = "/mnt/public03/usr/sunyifei2/engagement_bait_datasets/20241203/dapan_video_info.csv"
 
    model_name='Bert'     
    batch_size = 128    

    path_datasets = None    
    mode='test'
    distributed =False
    world_size = 4   
    class_file_path='/mnt/public03/usr/sunyifei2/risk_text_classifier/configfile/class.txt'
    path_model_save="/mnt/public03/usr/sunyifei2/risk_text_classifier/logs/SunyifeiBertConfig/20241129_165131/checkpoints"
    which_step='step_best'


