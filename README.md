# RUAS

this is the official code for the paper "Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement"

## Environment Preparing
```
python 3.6
pytorch 0.4.1
```

### Testing

We provide different models which are trained from different datasets.
*lol* is trained from LOL dataset.
*upe* is trained from MIT5K dataset.
*dark* is trained from DarkFace dataset.
Finally, run *test.py*, the results will be saved in `./result/`
```
python test.py 
--data_path           #The folder path of the picture you want to test
E:/test/
--model               #The checkpoint name
lol or upe or dark
--save_path            #The save path of the picture processed
./result/
```

### Training

If you want to train your own model on a new dataset, run *train.py*. 
Only low light images are needed.
The model will be saved in `./EXP/train/weights.pt`
```
python train.py 
```

### Searching
Please get train set and valid set ready, and run *train_search.py*.
Due to the data you used is different from ours, it is reasonable that the searched architecture is different from ours.
```
python train_search.py 
```


### Reference

If you find our work useful in your research please consider citing our paper:
```
@inproceedings{liu2021ruas,
title = {Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement},
author = {Risheng, Liu and Long, Ma and Jiaao, Zhang and Xin, Fan and Zhongxuan, Luo},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year = {2021}
}
```

A great thanks to [DARTS](https://github.com/quark0/darts) for providing the basis for this code.
