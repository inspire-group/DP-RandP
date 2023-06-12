####Example for configuration of experiment.
N=100
BS=50000
eps=1
sigma=38
seed=11297
feat_root="/data/xinyut/randp/features/"
data_root="/data/xinyut/datasets/cifar10"
ckpt_root="/data/xinyut/randp/ckpt/"
for lr in 10
do
    python cifar.py --seed $seed --lr $lr --epochs $N --batch_size $BS --epsilon $eps --sigma $sigma --random_dataset "stylegan-oriented" --ckpt_root $ckpt_root --data_root $data_root  --feat_root $feat_root
done