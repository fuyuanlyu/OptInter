 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memorize-factorize-or-be-naive-learning/click-through-rate-prediction-on-avazu)](https://paperswithcode.com/sota/click-through-rate-prediction-on-avazu?p=memorize-factorize-or-be-naive-learning)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memorize-factorize-or-be-naive-learning/click-through-rate-prediction-on-ipinyou)](https://paperswithcode.com/sota/click-through-rate-prediction-on-ipinyou?p=memorize-factorize-or-be-naive-learning)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memorize-factorize-or-be-naive-learning/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=memorize-factorize-or-be-naive-learning) 

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

````
@inproceedings{OptInter,
  author       = {Fuyuan Lyu and
                  Xing Tang and
                  Huifeng Guo and
                  Ruiming Tang and
                  Xiuqiang He and
                  Rui Zhang and
                  Xue Liu},
  title        = {Memorize, Factorize, or be Naive: Learning Optimal Feature Interaction
                  Methods for {CTR} Prediction},
  booktitle    = {38th {IEEE} International Conference on Data Engineering, {ICDE} 2022},
  pages        = {1450--1462},
  address      = {Kuala Lumpur, Malaysia},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/ICDE53745.2022.00113},
  doi          = {10.1109/ICDE53745.2022.00113},
  timestamp    = {Sun, 30 Jul 2023 12:39:38 +0200},
  biburl       = {https://dblp.org/rec/conf/icde/LyuTGTHZL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
````