# stock_predict_CNN_LSTM

## installaction(req_)

After cloning the repository, the package can be installed from inside the main directory with

```sh
pip install -r .\req_\requirements.txt
```

如果遇到安裝TA-Lib套件時候遇到error: Microsoft Visual C++ 14.0 is required.

```sh
pip install .\req_\TA_Lib-0.4.21-cp38-cp38-win_amd64.whl
```

上列.whl 基於python 3.8
可根據python 版本安裝對應包
下載連結: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

如果安裝完成遇到仍遇到import talib error
```sh
pip install numpy --upgrade
```
請確認升級numpy >= 1.20.0

The code has run at some point with Python 3.6. and 3.8.

# Model

### CNN8
```
python .\CNN8_class3.py
```
可以建立出CNN8 model 並存放於mlruns/0/<ModelID>
同儲存位置亦含有模型loss/accuarcy/confusion.png

## experiments
實踐進程與歷史紀錄 可能會有資料集合位置問題須注意更改
class2 內 .ipynb 為CNN 2分類實驗進程可參照
class3 內 .ipynb 為3分類 CNN/LSTM 比較與優化實驗進程可參照

| Model Name      | Input Variables     |
| ---------- | :-----------:  |
| CNN1    | Closing price     |
| CNN2    | Closing price, SMA, EMA     |
| CNN3    | Closing price, SMA, EMA, ROC, MACD     |
| CNN4, LSTM4    | Closing Price, SMA, EMA, ROC, MACD, Fast %K, Slow %D, Upper Band, Lower Band、%B     |
| CNN5    | Closing price、石油價格、石油波動指數     |
| CNN6    | Closing price、黃金價格、黃金波動指數     |
| CNN7, LSTM7    | Closing price、石油價格、石油波動指數、黃金價格、黃金波動指數     |
| CNN8    | Closing Price, SMA, EMA, ROC, MACD, Fast %K, Slow %D, Upper Band, Lower Band、%B、石油價格、石油波動指數、黃金價格、黃金波動指數     |

## Data

資料集來源: https://finance.yahoo.com/

以下說明各檔案資料集與資料區間：

| 檔案名稱      | 資料集     | 區間     | 資料起始日 | 資料結束日 |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: |
| 1019_SP.cvs     | S&P500     | 每日     | 2010/01/01  | 2019/12/31 |
| 1019_GC.cvs    | 黃金期貨價格     | 每日     | 2010/01/01  | 2019/12/31 |
| 1019_CL.cvs     | 原油期貨價格     | 每日     | 2010/01/01  | 2019/12/31 |
| 1019_USO.cvs     | 美國石油指數基金     | 每日     | 2010/01/01  | 2019/12/31 |
  
## Cite the paper
----
If this repository, the paper or any of its content is useful for your research, please cite:
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
