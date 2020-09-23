import os
import pandas as pd
from flask import Flask, request
from ashare_price import download_price
from ashare_foundamentals import download_foundamental
from preprocess_predict import preprocess_and_predict

app = Flask(__name__)

def download_data(a_ticker):
    df_tickers = pd.read_excel(r"Tickers_download.xlsx")
    if a_ticker in df_tickers["Ticker"].values.tolist():
        print("data already existed;")
    else:
        print("start to download data;")
        df_tickers.loc[len(df_tickers)] = a_ticker
        df_tickers.to_excel("Tickers_download.xlsx", index=False, encoding='utf-8')
        download_price(1, len(df_tickers), "stock")
        os.system('merge_price.sh')
        for i in ["gro", "pro", "bal"]:
            download_foundamental(len(df_tickers)-1, len(df_tickers), i)
            os.system('merge_{}.sh'.format(i))

@app.route("/predict_stock/", methods=["GET"])
def predict_stock():
    req_data = request.args.get("ticker")
    download_data(req_data)
    response = preprocess_and_predict(req_data)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug=True)