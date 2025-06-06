python src/main.py mtable val \
    --model src/cfg/models/startsr-mtable.yaml \
    --source datasets/ICDAR2013/test_images \
    --model_path checkpoints/ICDAR2013/star/model_best.pth \
    --label datasets/ICDAR2013/labels/test.json \
    --device 0,1 \
    --save_result \
    --infer_workers 12 \
    --project Val/ICDAR2013 \
    --name star