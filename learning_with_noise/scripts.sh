DATAPATH="/data/xinyut/randp/data"
ckpt_path="/data/xinyut/randp/ckpt"
python align_uniform/main.py --imagefolder $DATAPATH/stylegan-oriented --result $ckpt_path/stylegan-oriented --gpus 0 0 --depth 16 --width 4