DATA=DB15K
EMB_DIM=250
NUM_BATCH=1024
MARGIN=12
LR=2e-5
NEG_NUM=128
EPOCH=1000

CUDA_VISIBLE_DEVICES=0 nohup python run_adamf_mat.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=$EPOCH \
  -dim=$EMB_DIM \
  -mu=0 \
  -save=./checkpoint/$DATA-$NUM_BATCH-$EMB_DIM-$NEG_NUM-$MARGIN-$LR-$EPOCH \
  -neg_num=$NEG_NUM \
  -learning_rate=$LR > $DATA-$EMB_DIM-$NUM_BATCH-$NEG_NUM-$MARGIN-$EPOCH.txt &
