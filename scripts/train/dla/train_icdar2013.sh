mkdir -p ../Train/ICDAR2013/dla/weights
cp ../checkpoints/WTW/dla/model_last.pth ../Train/ICDAR2013/dla/weights/
# Fine-tune for 100 epochs, starting from the 200th.
python main.py mtable train \
    --model cfg/models/dlatsr-mtable.yaml \
    --data cfg/datasets/ICDAR2013.yaml \
    --epochs 300 \
    --device 0,1 \
    --master_batch -1 \
    --batch 22 \
    --workers 32 \
    --lr_step 270,290 \
    --val_epochs 10 \
    --project Train/ICDAR2013 \
    --name dla \
    --resume