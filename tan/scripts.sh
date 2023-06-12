####epsilon=1, sigma=9.3 for 875 steps.
eps=1
steps=875
noise=9.3
ft_epoch=8
lplr=15
lr=0.4
seed=9

python cifar10.py --experiment $seed --ft_epoch $ft_epoch --lp_lr $lplr --batch_size 4096 --transform 16 --epsilon $eps --max_nb_steps $steps --sigma $noise --lr $lr --max_physical_batch_size 4096 --load_pretrained --pretrained_path "/scratch/gpfs/xinyut/new_encoders/wrn/" --data_root /scratch/gpfs/xinyut/datasets/cifar10 --dump_path "/scratch/gpfs/xinyut/tmp"
