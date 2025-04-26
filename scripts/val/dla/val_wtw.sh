python src/main.py mtable val \
    --model src/cfg/models/dlatsr-mtable.yaml \
    --source datasets/WTW/test_images \
    --model_path checkpoints/WTW/dla/model_best.pth \
    --label datasets/WTW/labels/test.json \
    --device 0,1 \
    --save_result \
    --infer_workers 12 \
    --project Test/WTW \
    --name dla