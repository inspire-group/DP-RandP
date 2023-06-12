This repo is built based on [source repo](https://github.com/mbaradad/learning_with_noise) that contains the code for paper: 

> [**Learning to See by Looking at Noise**]  (NeurIPS 2021, Spotlight)
>
> [[Project page](https://mbaradad.github.io/learning_with_noise/)]  [[Paper](https://arxiv.org/pdf/2106.05963.pdf)] [[arXiv](https://arxiv.org/abs/2106.05963)]
>
> Manel Baradad and Jonas Wulff and Tongzhou Wang and Phillip Isola and Antonio Torralba.

For more details, please check the source repo.

## Prepare Synthetic Data
We mainly use the `stylegan-oriented' images (other datases are also available) provided by the source repo. You can also refer to [Data Generation](https://github.com/mbaradad/learning_with_noise#data-generation) in source repo to generate synthetic images.

```
DATAPATH="/data/xinyut/randp/data"
wget -O $DATAPATH/stylegan-oriented.zip http://data.csail.mit.edu/noiselearning/zipped_data/small_scale/stylegan-oriented.zip
unzip $DATAPATH/stylegan-oriented.zip -d $DATAPATH/stylegan-oriented
```

## Training the model

Then you can launch the contrastive training in our Phase I with:
```
DATAPATH="/data/xinyut/randp/data"
ckpt_path="/data/xinyut/randp/ckpt"
python align_uniform/main.py --imagefolder $DATAPATH/stylegan-oriented --result $ckpt_path/stylegan-oriented --gpus $GPU_ID0 $GPU_ID1
```
Note that the contrastive training requires GPU_ID0 and GPU_ID1 each provides at least 4GB of memory for WRN16-4. You can use the same GPU for $GPU_ID0 and $GPU_ID1 if a single GPU has enough memotry.

## Citation
```
@inproceedings{baradad2021learning,
  title={Learning to See by Looking at Noise},
  author={Manel Baradad and Jonas Wulff and Tongzhou Wang and Phillip Isola and Antonio Torralba},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```