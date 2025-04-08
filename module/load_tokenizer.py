from module.ModelMap import map_tokenizer
from module.tokenizer.TextTokenizer import TextTokenizer
from module.tokenizer.LMTextTokenizer import LMTextTokenizer
import os


def load_tokenizer(config):
    """
    读取分词器
    """
    print('loading tokenizer config ...')
    tokenizer = map_tokenizer(config.model_name)
    if not tokenizer:
        print('toknizer {} is null, please check your model name.'.format( config.model_name))

    if  config.model_name not in  config.lm_model_list:
        path_token = os.path.join( config.path_datasets, 'vocab.pkl')
        tokenizer = tokenizer()
        # 若存在词表，则直接读取
        if os.path.exists(path_token):
             tokenizer.load(path_token)
        else:
            # 否则读取训练数据，并创建词表

            path_corpus = os.path.join( config.path_datasets, '{}.csv'.format( config.train_dataset))
            corpus = pd.read_csv(path_corpus)['text'].tolist()
            # corpus, _ = open_file(path_corpus, sep='\t')
            token2index, _ =  tokenizer.create(corpus)
            # 标签映射表存到本地
            write_file(token2index, path_token + '.txt')
            pkl.dump(token2index, open(path_token, 'wb'))
            tokenizer.load(path_token)
    else:
        print(f"load tokenizer.from_pretrained {config.initial_pretrain_tokenizer}")
        tokenizer = tokenizer.from_pretrained(config.initial_pretrain_tokenizer)
        tokenizer = LMTextTokenizer(tokenizer,config)
    print('Vocab size: {}'.format(len( tokenizer.token2index)))

    return tokenizer