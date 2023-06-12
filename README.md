# DP-RandP

> Differentially Private Image Classification by Learning Priors from Random Processes [[arxiv](https://arxiv.org/abs/2306.06076)]
>
> Xinyu Tang*, Ashwinee Panda*, Vikash Sehwag, Prateek Mittal (*: equal contribution)
>
>


## Requirements
This version of code has been tested with Python 3.9.16 and PyTorch 1.12.1. 

Set up environment via pip and anaconda
```
conda create -n "dprandp" python=3.9 
conda activate dprandp
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r ./requirements.txt
```

Note that we made small edits in [opacus/data_loader.py](https://github.com/pytorch/opacus/blob/3a7e8f82a8d02cc1ed227f2ef287865d904eff8d/opacus/data_loader.py#L198) to make sure the expected batch size of poisson sampling is the same as the hyperparameters in the code instead of approximated by 1 / len(data_loader) to be consistant with privacy accounting.
```
sample_rate=data_loader.batch_size/len(data_loader.dataset), #instead of 1 / len(data_loader)
```

We also provide the corresponding data_loader.py we used in `./opacus_utils` for reference. 
```
cp ./opacus_utils/data_loader.py $YOUR_opacus_lib_path/data_loder.py
```

We also make change accordingly in ./tan/src/opacus_augmented/privacy_engine_augmented.py (line 386-387).

Please make change accordingly for `--max_physical_batch_size` to fit your GPU memory.

## Experiment
For Phase I, please refer to folder `./learning_with_noise`. 

For Phase II and Phase III, please refer to folder `./tan`. 

For Table 4 in paper for DP-RandP w/o Phase III, please refer to foloder `./linear_prob`.

## Citation
```
@article{tang2023dprandp,
      title={Differentially Private Image Classification by Learning Priors from Random Processes}, 
      author={Xinyu Tang and Ashwinee Panda and Vikash Sehwag and Prateek Mittal},
      journal={arXiv preprint arXiv:2306.06076},
      year={2023}
}
```

## Credits  
This code has been built upon the code accompanying the papers

"[Learning to See by Looking at Noise](https://arxiv.org/abs/2106.05963)" [[code](https://github.com/mbaradad/learning_with_noise)].

"[TAN Without a Burn: Scaling Laws of DP-SGD](https://arxiv.org/abs/2210.03403)" [[code](https://github.com/facebookresearch/tan)].

The hyperparameter setting of the code mostly follows the papers

"[Unlocking High-Accuracy Differentially Private Image Classification through Scale](https://arxiv.org/abs/2204.13650)" [[code](https://github.com/deepmind/jax_privacy)].

"[A New Linear Scaling Rule for Differentially Private Hyperparameter Optimization](https://arxiv.org/abs/2212.04486)" [[code](https://github.com/kiddyboots216/dp-custom)].



## Questions
If anything is unclear, please open an issue or contact Xinyu Tang (xinyut@princeton.edu).
