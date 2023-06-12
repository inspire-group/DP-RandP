This repo is built based on [source repo](https://github.com/facebookresearch/tan) that contains the code for paper: 

> [**TAN Without a Burn: Scaling Laws of DP-SGD**] (ICML 2023)
> 
> [[arxiv](https://arxiv.org/abs/2210.03403)]
>
> Sander, Tom and Stock, Pierre, and Sablayrolles, Alexandre.

For more details, please check the source repo.


### WideResNet on CIFAR-10
For all commands, set `max_physical_batch_size` to something that fits your GPU memory.

After Phase I and saving WRN16-4 encoder to `$ckpt_path`, can train model for Phase II and Phase II by the following command:
```
ckpt_path="/data/xinyut/randp/ckpt"
python cifar10.py --experiment 0 --ft_epoch 8 --lp_lr 15  --lr $lr --epsilon $eps --sigma $noise --max_nb_steps $steps --batch_size 4096 --transform 16  --max_physical_batch_size 4096 --load_pretrained --pretrained_path "$ckpt_path/wrn164/" --data_root $CIFAR10_DATAPATH

```
Note that ``--ft_epoch 0`` is equal to DP-RandP w/o Phase II. Set ``--ft_epoch`` exceeding the ``--max_nb_steps`` is equal to DP-RandP w/o Phase III.


## Citation
```
@article{sander2022tan,
  title={TAN Without a Burn: Scaling Laws of DP-SGD},
  author={Sander, Tom and Stock, Pierre, and Sablayrolles, Alexandre},
  journal={arXiv preprint arXiv:2210.03403},
  year={2022}
}
```

## License
This code is released under BSD-3-Clause, as found in the [LICENSE](https://github.com/facebookresearch/tan/LICENSE) file.
