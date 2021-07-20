CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py --config ./config/nturgbd-cross-subject/train_joint.yaml
CUDA_VISIBLE_DEVICES=2,3 nohup python -u main.py --config ./config/uav/train_joint.yaml >logs/random_sample.log 2>&1 &
tail -f logs/random_sample.log

CUDA_VISIBLE_DEVICES=0,2,3 nohup python -u main.py >logs/random_sample.log 2>&1 &
tail -f logs/random_sample.log

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py >logs/3-fold.log 2>&1 &
tail -f logs/3-fold.log