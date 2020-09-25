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


def download_foundamental(start, end, gpb):
    lg = bs.login()
    logger.info("login error code: {}, logging error messge: {}".format(lg.error_code, lg.error_msg))

    df_stock = pd.read_excel(r'Tickers_download.xlsx')  
    new_stock_list = []
    for i in df_stock.index:
        new_stock_list.append(df_stock['Ticker'][i])
        downloads = os.listdir('downloads_{}/'.format(gpb))
    had = set()
    for f in downloads:
        parts = f.split("_")
        had.add(int(parts[0]))

    t = start
    total = end - start
    count = 0
    for i in new_stock_list[start:end]:
        count += 1
        result_list = []
        for k in range(2015,2020):
            for j in range(1,5):
                if (t in had):
                    logger.info("====omit {}====".format(t))
                    t += 1
                    continue
                else:
                    logger.info("====process {}_{}, {} out of {}====".format(i, t, count, total))
                if gpb == "gro":
                    rs = bs.query_growth_data(code=i, year=k, quarter=j)
                elif gpb == "pro":
                    rs = bs.query_profit_data(code=i, year=k, quarter=j)
                elif gpb == "bal":
                    rs = bs.query_balance_data(code=i, year=k, quarter=j)
                logger.info("error code: {}, error message: {}".format(rs.error_code, rs.error_msg))
                
                while (rs.error_code == '0') & rs.next():
                    result_list.append(rs.get_row_data())
                # print(result_list)
        result = pd.DataFrame(result_list, columns=rs.fields)
        output = "downloads_{}/{}_{}.csv".format(gpb, t, i)
        t += 1
        result.to_csv(output, encoding="gbk", index=False)

    # print('login respond error_code:'+lg.error_code)
    # print('login respond  error_msg:'+lg.error_msg)

    # bs.logout()


if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    gpb = sys.argv[3]
    download_foundamental(start, end, gpb)
