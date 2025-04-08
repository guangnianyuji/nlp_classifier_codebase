set -e
WORK_DIR="/mnt/public03/usr/sunyifei2/nlp_codebase"
cd $WORK_DIR
echo $PWD
# pip install --upgrade pip  
# pip install setuptools==69.0.0
# pip config set global.index-url http://pypi.devops.xiaohongshu.com/simple/
# pip config set install.trusted-host pypi.devops.xiaohongshu.com
# pip install pandas==1.4.2 pyarrow #snownlp
# pip install accelerate==0.9.0 s3fs huggingface-hub==0.0.12 
# pip install  'transformers>=4.38'  
# ################################################################################
# pip install pandas==1.4.2 --upgrade
# pip install tqdm modelscope sentencepiece
# pip uninstall peft -y
# pip install peft 
# pip uninstall safetensors -y
# pip install safetensors 
# pip install onnxruntime onnxmltools onnxconverter_common
# pip install  'transformers==4.9.2'  
# pip install 'datasets==2.14.4'

# cd /mnt/nlp01/usr/lishanglin/code/level_classification_newstd/


# mkdir -p /mnt/nlp01/usr/lishanglin/code/level_classification_newstd/checkpoints/focal_denoise_3fc_notev2_sliding_tifeng/
# CUDA_VISIBLE_DEVICES=0 python3 -u main_tifeng.py --config FocalDenoise3FCNotev2SlidingTifengConfig --path_model_save checkpoints/focal_denoise_3fc_notev2_sliding_tifeng/ --batch_size 8 --path_datasets datasets/0802  2>&1 | tee /mnt/nlp01/usr/lishanglin/code/level_classification_newstd/checkpoints/focal_denoise_3fc_notev2_sliding_tifeng/output.txt
CONFIGNAME=SunyifeiXLMRobertaTest
#tag=$(date -d '+8 hours' +%Y%m%d_%H%M%S)
tag=$(date +%Y%m%d_%H%M%S)
mkdir -p $WORK_DIR/logs/embedding/$CONFIGNAME/
OUTPUT_DIR=$WORK_DIR/logs/embedding/$CONFIGNAME/$tag

mkdir -p $OUTPUT_DIR
LOG_PATH=$OUTPUT_DIR/log_output.txt

#CHECKPOINT_PATH=""
# MODEL_SAVE_PATH=$OUTPUT_DIR/checkpoints
# mkdir -p $MODEL_SAVE_PATH

# DATA_DIR='/mnt/public03/usr/sunyifei2/datasets/20250206_20250119tt_en_3236614/datasets'
# SAVE_DIR='/mnt/public03/usr/sunyifei2/datasets/20250206_20250119tt_en_3236614/datasets_result'
# STEP=$2
# VAL_DATASET=$DATA_DIR/data_$STEP.csv
# PATH_OUTPUT_FILE=$SAVE_DIR/result_$STEP.csv

# id=$1

CUDA_VISIBLE_DEVICES=0  python main.py --config $CONFIGNAME  --mode embedding  2>&1 | tee $LOG_PATH
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29507 infer.py --config $CONFIGNAME  --path_model_save $MODEL_SAVE_PATH  --distributed True 2>&1 | tee $LOG_PATH
# mkdir -p /mnt/nlp01/usr/lishanglin/code/level_classification_newstd/checkpoints/focal_denoise_3fc_notev2_sliding_tifeng2/
# CUDA_VISIBLE_DEVICES=7 python infer.py --config $CONFIGNAME   2>&1 | tee $LOG_PATH