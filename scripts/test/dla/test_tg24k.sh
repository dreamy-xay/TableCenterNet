python src/main.py mtable predict \
    --model src/cfg/models/dlatsr-mtable.yaml \
    --source datasets/TG24K/test_images \
    --model_path checkpoints/TG24K/dla/model_best.pth \
    --resolution 768 \
    --device 0,1 \
    --save \
    --save_result \
    --save_corners \
    --workers 12 \
    --project Test/TG24K \
    --name dla