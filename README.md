# OptInter
This repository contains PyTorch Implementation of ICDE 2022 paper: 

*Memorize, factorize, or be naive: Learning optimal feature interaction methods for CTR Prediction*. 



You can find our paper [here](https://arxiv.org/abs/2108.01265).



## Usage

Following the steps below to run our codes:

### Install

`pip install -r requirements.txt`



### Download Dataset

Please Download the [Criteo](https://www.kaggle.com/mrkmakr/criteo-dataset) and [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data) datasets. 

- For Criteo dataset, copy the `train.txt` file under `datasets/Criteo` and rename it to `full.txt`. 
- For Avazu dataset, copy the `train.csv` file under `datasets/Avazu` and rename it to `full.csv`.



### Preprocess Dataset

- For Criteo dataset, run `python preprocess/criteo.py`
- For Avazu dataset, run `python preprocess/avazu.py`



### Search

- For Criteo dataset, run `python learn/CriteoSearch.py`
- For Avazu dataset, run `python learn/AvazuSearch.py`



### Re-train

- For Criteo dataset, run `python learn/CriteoTrain.py --load XXX --model DNN_cart --alpha_mode 0`
- For Avazu dataset, run `python learn/AvazuTrain.py --load XXX --model DNN_cart --alpha_mode 0`

Here `XXX` indicates the logs dictionary generated during the search stage.



### Baseline Running

- For Criteo dataset, run `python learn/CriteoTrain.py --model YYY`
- For Avazu dataset, run `python learn/AvazuTrain.py --model YYY`

Here `YYY` could be `{LR, FM, FNN, IPNN, DeepFM, PIN, Poly2, DNN_cart}`



## Citation

```
​```
@article{lyu2021memorize,
  title={Memorize, Factorize, or be Na$\backslash$" ive: Learning Optimal Feature Interaction Methods for CTR Prediction},
  author={Lyu, Fuyuan and Tang, Xing and Guo, Huifeng and Tang, Ruiming and He, Xiuqiang and Zhang, Rui and Liu, Xue},
  journal={arXiv preprint arXiv:2108.01265},
  year={2021}
}
​```
```