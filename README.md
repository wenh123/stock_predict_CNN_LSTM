# stock_predict_CNN_LSTM

## installaction(req_)

After cloning the repository, the package can be installed from inside the main directory with

```sh
pip install -r .\req_\requirements.txt
```

If you encounter an error: Microsoft Visual C++ 14.0 is required when installing the TA-Lib package.


```sh
pip install .\req_\TA_Lib-0.4.21-cp38-cp38-win_amd64.whl
```

The above .whl is based on python 3.8
The corresponding package can be installed according to the python version
download link: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

If you encounter `import talib error` during installation
```sh
pip install numpy --upgrade
```
Please upgrade numpy >= 1.20.0

The code has run at some point with Python 3.6. and 3.8.

# Model

### CNN8
```
python .\CNN8_class3.py
```

CNN8 model can be created and stored in mlruns/0/<ModelID>
The same storage location also contains the model loss/accuarcy/confusion.png
  
## experiments

The practice process and historical records may have data collection location problems, which must be changed.
The CNN 2 classification experiment process can refer to experiments/class2.
The 3-class classification with CNN/LSTM comparison and optimization experiment process can refer to experiments/class3.

| Model Name      | Input Variables     |
| ---------- | :-----------:  |
| CNN1    | Closing price     |
| CNN2    | Closing price, SMA, EMA     |
| CNN3    | Closing price, SMA, EMA, ROC, MACD     |
| CNN4, LSTM4    | Closing Price, SMA, EMA, ROC, MACD, Fast %K, Slow %D, Upper Band, Lower Band、%B     |
| CNN5    | Closing price, Oil price, Oil volatility index     |
| CNN6    | Closing price, Gold price, Gold volatility index     |
| CNN7, LSTM7    | Closing price, Oil price, oil volatility index, gold price, gold volatility index     |
| CNN8    | Closing Price, SMA, EMA, ROC, MACD, Fast %K, Slow %D, Upper Band, Lower Band、%B, Oil price, oil volatility index, gold price, gold volatility index    |

## Data

Data Source: https://finance.yahoo.com/

The following describes each file data set and data interval:

| File name | Data set | Interval | Data start date | Data end date |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: |
| 1019_SP.cvs | S&P500 | Daily | 2010/01/01 | 2019/12/31 |
| 1019_GC.cvs | Gold Futures Price | Daily | 2010/01/01 | 2019/12/31 |
| 1019_CL.cvs | Crude Oil Futures Prices | Daily | 2010/01/01 | 2019/12/31 |
| 1019_USO.cvs | US Oil Index Fund | Daily | 2010/01/01 | 2019/12/31 |
  
## Cite the paper

If this repository, the paper or any of its content is useful for your research, please cite:
```  
@article{CHEN2021107760,
title = {Constructing a stock-price forecast CNN model with gold and crude oil indicators},
journal = {Applied Soft Computing},
volume = {112},
pages = {107760},
year = {2021},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2021.107760},
url = {https://www.sciencedirect.com/science/article/pii/S1568494621006815},
author = {Yu-Chen Chen and Wen-Chen Huang},
keywords = {Stock price forecast, Deep learning, Convolutional neural networks, Long short-term memory, Bayesian optimization},
}
```
