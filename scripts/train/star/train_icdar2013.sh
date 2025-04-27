mkdir -p ../Train/ICDAR2013/star/weights
cp ../checkpoints/WTW/star/model_last.pth ../Train/ICDAR2013/star/weights/
# Fine-tune for 100 epochs, starting from the 200th.
python main.py mtable train \
    --model cfg/models/startsr-mtable.yaml \
    --data cfg/datasets/ICDAR2013.yaml \
    --epochs 300 \
    --device 0,1 \
    --master_batch -1 \
    --batch 22 \
    --workers 32 \
    --lr_step 270,290 \
    --val_epochs 10 \
    --project Train/ICDAR2013 \
    --name star \
    --resume