import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np

yf.pdr_override()

df = pd.read_excel(r'D:\sandp500\NASDAQlist.xlsx', sheet_name='Sheet 1')

new_stock_list = []
for i in df.index:
    print(df['Ticker'][i])
    new_stock_list.append(df['Ticker'][i])

for j in new_stock_list:
    df = pdr.get_data_yahoo(tickers=j, start='2007-01-01', end='2018-12-31', interval="1d")
    df.dropna(subset=['Volume'], inplace=True)
    # df.insert(0, "Name", [j], True)
    print(df)
    df_dict = df.to_dict('list')
    df_dict["Name"] = j
    print(df_dict)
    df_new = pd.DataFrame(df_dict, columns=['Name', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
    print(df_new)
    output_name = j + '_data.csv'
    csv_path = r'.\individual_stocks_11yr\{}'.format(output_name)
    df_list = pd.DataFrame(df_new)
    df_new.to_csv(csv_path, index=True)

