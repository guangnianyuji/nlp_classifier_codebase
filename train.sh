set -e
WORK_DIR="/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase"
cd $WORK_DIR
echo $PWD
# pip install --upgrade pip  
# pip install setuptools==69.0.0
# pip config set global.index-url http://pypi.devops.xiaohongshu.com/simple/
# pip config set install.trusted-host pypi.devops.xiaohongshu.com
# pip install pandas pyarrow #snownlp
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

CONFIGNAME=SunyifeiMultiTanghuang48LConfig
tag=$(date +%Y%m%d_%H%M%S)
#tag=$(date +%Y%m%d_%H%M%S)
mkdir -p $WORK_DIR/logs/train/$CONFIGNAME/

OUTPUT_DIR=$WORK_DIR/logs/train/$CONFIGNAME/$tag
mkdir -p $OUTPUT_DIR

LOG_PATH=$OUTPUT_DIR/log_output.txt

MODEL_SAVE_PATH=$OUTPUT_DIR/checkpoints
mkdir -p $MODEL_SAVE_PATH

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nproc_per_node=8 --master_port=29507 main.py --dir $OUTPUT_DIR --distributed --mode train --config $CONFIGNAME --path_model_save $MODEL_SAVE_PATH  --train_dataset /mnt/nj-larc/usr/ajie1/sunyifei/datasets/20250321_1kw/all_file --val_dataset /mnt/nj-larc/usr/ajie1/sunyifei/datasets/20250321_group_eval/data_88112.csv  2>&1 | tee $LOG_PATH

# --resume_from /mnt/public03/usr/sunyifei2/risk_text_classifier/logs/SunyifeiXLMRoberta/20250122_164928/checkpoints/step_10000