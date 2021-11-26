# OptInter
This repository contains PyTorch Implementation of ICDE submission paper: Memorize, factorize, or be naive: Learning optimal feature interaction methods for CTR Prediction. 



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

