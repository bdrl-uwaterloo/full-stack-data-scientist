# 0713 modified by BZ

import baostock as bs
import pandas as pd
import sys
import os
import logging

logFormatter = logging.Formatter("%(asctime)s [%(name)3s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def download_price(start, end, ind_or_sto):
    # print(start, end)
    # login system
    lg = bs.login()
    # Display login infomration
    logger.info("login error code: {}, logging error messge: {}".format(lg.error_code, lg.error_msg))

    # Login error messgae, a detialed explanation of the error.
    df_stock = pd.read_excel(r'Tickers_download.xlsx')  # , sheet_name='Sheet 1'
    new_stock_list = []
    for i in df_stock.index:
        # print(df_stock['Ticker'][i])
        new_stock_list.append(df_stock['Ticker'][i])
    # print(new_stock_list)

    downloaded = os.listdir('downloads_price/')
    had = set()
    for f in downloaded:
        parts = f.split("_")
        had.add(int(parts[0]))

    t = start
    total = end - start
    count = 0
    for i in new_stock_list[start:end]:
        count += 1
        if (t in had):
            logger.info("====omit {}====".format(t))
            t += 1
            continue
        else:
            logger.info("====process {}_{}, {} out of {}====".format(i, t, count, total))
        result_list = []
        if ind_or_sto == "index":
            rs = bs.query_history_k_data_plus(i, "date,code,open,high,low,close,volume,amount", frequency="d")
        elif ind_or_sto == "stock":
            rs = bs.query_history_k_data(i, "date,code,open,high,low,close,volume,amount,adjustflag,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,pctChg,isST", frequency="d", adjustflag="2")
        logger.info("error code: {}, error message: {}".format(rs.error_code, rs.error_msg))

        while (rs.error_code == '0') & rs.next():

            result_list.append(rs.get_row_data())
        result = pd.DataFrame(result_list, columns=rs.fields)
        output = "downloads_price/{}_{}.csv".format(t, i)
        t += 1
        result.to_csv(output, encoding="gbk", index=False)

    # print('login respond error_code:'+lg.error_code)
    # print('login respond  error_msg:'+lg.error_msg)

    # log out
    # bs.logout()


if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    ind_or_sto = sys.argv[3]
    download_price(start, end, ind_or_sto)
