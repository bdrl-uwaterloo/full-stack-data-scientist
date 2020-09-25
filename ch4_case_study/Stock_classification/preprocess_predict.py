import os
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def add_ma_column(ma_start, k, ma, df_tmp):
    close_list = df_tmp['close'].to_list()
    ma_list = []
    for j in range(ma_start, len(close_list)):
        tmp_list = close_list[int(j-k):int(j)]
        avrg = numpy.mean(tmp_list)
        ma_list.append(avrg)
    return ma_list

def add_vma_column(vma_start, k, vma, df_tmp):
    amount_list = df_tmp['amount'].to_list()
    vma_list = []
    for j in range(vma_start, len(amount_list)):
        tmp_list = amount_list[int(j-k):int(j)]
        avrg = numpy.mean(tmp_list)
        vma_list.append(avrg)
    return vma_list

def preprocess():
    # create dataframes
    csi300_price = pd.read_csv('0_sh.000300.csv', index_col="date", parse_dates=True, encoding='gbk')
    daily_97_price = pd.read_csv('daily_price.csv', index_col="date", parse_dates=True, encoding='gbk')
    quarter_all = pd.read_csv('quarterly_pro.csv', encoding='gbk')
    quarterly_growth_97 = pd.read_csv('quarterly_gro.csv', encoding='gbk')
    quarterly_balance_97 = pd.read_csv('quarterly_bal.csv', encoding='gbk')

    # get all unique code
    daily_97_price_unique_code = daily_97_price.drop_duplicates(subset=['code'],keep='first')
    code_list = daily_97_price_unique_code['code'].to_list()
    print(len(code_list))

    df_columns = [
        'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag', 'turn', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
        'isST', 'ma10', 'ma20', 'ma30', 'ma60', 'vma5', 'vma60'
    ]

    # calculate ma5, ma10, ma20, ma30, ma60
    df_ma = pd.DataFrame(columns=df_columns)
    for i in code_list:
        df_tmp = daily_97_price[daily_97_price['code'].isin([i])]
        start = 60
        df_tmp_1 = df_tmp[int(start):]
        df_tmp_2 = df_tmp_1.copy()
      
        df_tmp_2['ma10'] = add_ma_column(start, 10, 'ma10', df_tmp)
        df_tmp_2['ma20'] = add_ma_column(start, 20, 'ma20', df_tmp)
        df_tmp_2['ma30'] = add_ma_column(start, 30, 'ma30', df_tmp)
        df_tmp_2['ma60'] = add_ma_column(start, 60, 'ma60', df_tmp)
        df_tmp_2['vma5'] = add_vma_column(start, 5, 'vma5', df_tmp)
        df_tmp_2['vma60'] = add_vma_column(start, 60, 'vma60', df_tmp)
        df_ma = df_ma.append(df_tmp_2)

    df_ma.to_csv('df_ma.csv', index=True, encoding='gbk')

    # set today's amount, turn, peTTM, pbMRQ to yesterday's value
    df_ma_1 = df_ma.copy()
    df_ma_2 = pd.DataFrame(columns=df_columns)
    # print(df_ma_1, "df_ma_1")
    for i in code_list:
        df_tmp = df_ma_1[df_ma_1['code'].isin([i])]
        # print(df_tmp, "df_tmp")
        amount_list = df_tmp['amount'].to_list()[0:-1]
        turn_list = df_tmp['turn'].to_list()[0:-1]
        peTTM_list = df_tmp['peTTM'].to_list()[0:-1]
        pbMRQ_list = df_tmp['pbMRQ'].to_list()[0:-1]
        df_tmp_1 = df_tmp.copy()
        # print(df_tmp_1, "df_tmp_1")
        df_tmp_1.drop(df_tmp_1.index[0],inplace=True)
        # print(df_tmp_1, len(df_tmp_1))
        df_tmp_1['amount'] = amount_list
        df_tmp_1['turn'] = turn_list
        df_tmp_1['peTTM'] = peTTM_list
        df_tmp_1['pbMRQ'] = pbMRQ_list
        df_ma_2 = df_ma_2.append(df_tmp_1)
    # print(df_ma_2)
    df_ma_2.to_csv('df_ma_2.csv', index=True, encoding='gbk')

    # when the market was closed, use tomorrow's price to fill stock price, and user yesteday's value to fill today's amount, turn, peTTM, pbMRQ
    df_ma_3 = df_ma_2.copy()
    start_date = str(df_ma_3.index[0])
    # print(df_ma_3.index[0])
    end_date = str(df_ma_3.index[-1])
    idx = pd.date_range(start_date, end_date)
    df_final = pd.DataFrame(columns=df_columns)
    for i in code_list:
        df_tmp = df_ma_3[df_ma_3['code'].isin([i])]
        df_tmp = df_tmp.reindex(idx)
        df_tmp[['amount','turn','peTTM','pbMRQ']] = df_tmp[['amount','turn','peTTM','pbMRQ']].ffill()
        df_tmp.bfill(inplace=True)
        df_final = df_final.append(df_tmp)
    df_final.index.name = 'date'
    # print(df_final)
    df_final.to_csv('df_final_bfill.csv', index=True, encoding='gbk')
    df_final_bfill = pd.read_csv('df_final_bfill.csv', encoding='gbk')

    # fill csi300 price when the market was closed
    csi300_final = pd.DataFrame(columns=df_columns)
    csi_tmp = csi300_price[csi300_price['code'].isin(['sh.000300'])]
    csi_start_date = str(csi300_price.index[0])
    csi_end_date = str(csi300_price.index[-1])
    csi_idx = pd.date_range(csi_start_date, csi_end_date)
    csi_tmp = csi_tmp.reindex(csi_idx)
    csi_tmp.bfill(inplace=True)
    csi300_final = csi300_final.append(csi_tmp)
    csi300_final.index.name = 'date'
    # print(csi300_final)
    csi300_final.to_csv('csi300_final_bfill.csv', index=True, encoding='gbk')
    csi300_final_bfill = pd.read_csv('csi300_final_bfill.csv', encoding='gbk')

    # Do NOT remove 2020 data
    quarter_growth = [
        'code', 'pubDate', 'statDate', 'currentRatio', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI'
    ]
    growth_set_1 = pd.DataFrame(columns=quarter_growth)
    for i in code_list:
        df_tmp = quarterly_growth_97[quarterly_growth_97['code'].isin([i])].copy()
        growth_set_1 = growth_set_1.append(df_tmp)
    growth_set_1.to_csv('growth_set_1.csv', index=False, encoding='gbk')

    balance_set = pd.DataFrame()
    balance_set['code'] = quarterly_balance_97['code'].to_list()
    balance_set['pubDate'] = quarterly_balance_97['pubDate'].to_list()
    balance_set['YOYLiability'] = quarterly_balance_97['YOYLiability'].to_list()
    balance_set['liabilityToAsset'] = quarterly_balance_97['liabilityToAsset'].to_list()
    balance_set['assetToEquity'] = quarterly_balance_97['assetToEquity'].to_list()
    balance_set_1 = pd.DataFrame(columns=['code','pubDate','YOYLiability','liabilityToAsset','assetToEquity'])
    for i in code_list:
        df_tmp = balance_set[balance_set['code'].isin([i])].copy()
        # print(df_tmp.tail(1))
        # df_tmp.drop(df_tmp.index[-1],inplace=True)
        balance_set_1 = balance_set_1.append(df_tmp)
    # print(balance_set_1)
    balance_set_1.to_csv('balance_set_1.csv', index=False, encoding='gbk')

    # drop duplicate pubDate
    quarter_columns = [
        'code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM',
        'MBRevenue', 'totalShare', 'liqaShare'
    ]
    quarter_final = pd.DataFrame(columns=quarter_columns)
    growth_set_2 = pd.DataFrame(columns=quarter_growth)
    balance_set_2 = pd.DataFrame(columns=['code','pubDate','YOYLiability','liabilityToAsset','assetToEquity'])
    for i in code_list:
        df_tmp = quarter_all[quarter_all['code'].isin([i])].copy()
        df_tmp = df_tmp.drop_duplicates(subset=['pubDate'], keep='last')
        quarter_final = quarter_final.append(df_tmp)
        df_tmp_1 = growth_set_1[growth_set_1['code'].isin([i])].copy()
        df_tmp_1 = df_tmp_1.drop_duplicates(subset=['pubDate'], keep='last')
        growth_set_2 = growth_set_2.append(df_tmp_1)
        df_tmp_2 = balance_set_1[balance_set_1['code'].isin([i])].copy()
        df_tmp_2 = df_tmp_2.drop_duplicates(subset=['pubDate'], keep='last')
        balance_set_2 = balance_set_2.append(df_tmp_2)
    # print(growth_set_2)
    # print(balance_set_2)

    # filter out 97 stocks from quarter_final
    index_1 = quarter_final['code'].isin(df_final_bfill['code'])
    quarter_97_1 = quarter_final[index_1]
    # print(quarter_97_1)

    # save the df to csv
    growth_set_2.to_csv('growth_set_2.csv', index=False, encoding='gbk')
    balance_set_2.to_csv('balance_set_2.csv', index=False, encoding='gbk')
    quarter_97_1.to_csv('quarter_97_1.csv', index=False, encoding='gbk')
    # quarter_97_2 = pd.read_csv('quarter_97_1.csv', encoding='gbk')
    quarter_97_1_1 = quarter_97_1.copy()
    quarter_97_1_1['YOYAsset'] = growth_set_2['YOYAsset'].to_list()
    quarter_97_1_1['YOYNI'] = growth_set_2['YOYNI'].to_list()
    quarter_97_1_1['YOYEPSBasic'] = growth_set_2['YOYEPSBasic'].to_list()
    quarter_97_1_1['YOYPNI'] = growth_set_2['YOYPNI'].to_list()
    quarter_97_1_1['YOYLiability'] = balance_set_2['YOYLiability'].to_list()
    quarter_97_1_1['liabilityToAsset'] = balance_set_2['liabilityToAsset'].to_list()
    quarter_97_1_1['assetToEquity'] = balance_set_2['assetToEquity'].to_list()
    # print(quarter_97_1_1)
    quarter_97_1_1.to_csv('quarter_97_1_1.csv', index=False, encoding='gbk')
    quarter_97_2 = quarter_97_1_1.copy()

    # combine code and date columns
    df_final_bfill['codeDate'] = df_final_bfill['date'] + df_final_bfill['code'] 
    quarter_97_2['codeDate'] = quarter_97_2['pubDate'] + quarter_97_2['code'] 
    # print(df_final_bfill)
    # print(quarter_97_2)

    # filter out 97 stocks quarter prices from df_ma_1
    index_2 = df_final_bfill['codeDate'].isin(quarter_97_2['codeDate'])
    quarter_97_price = df_final_bfill[index_2]
    # print(quarter_97_price)
    quarter_97_price.to_csv('quarter_97_price.csv', index=False, encoding='gbk')
    quarter_97_price_1 = pd.read_csv('quarter_97_price.csv', encoding='gbk')

    # find csi300 quarter price for each stock
    csi300_quarter = pd.DataFrame(columns=df_columns)
    for i in code_list:
        quarter_97_tmp = quarter_97_price_1[quarter_97_price_1['code'].isin([i])]
        csi300_tmp = csi300_final_bfill[csi300_final_bfill['date'].isin(quarter_97_tmp['date'])]
        csi300_quarter = csi300_quarter.append(csi300_tmp)
    csi300_quarter.to_csv('csi300_quarter.csv', index=False, encoding='gbk')

    # combine quarter_97_price and quarter_97_2
    combine_columns = [
        'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 
        'liqaShare', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'YOYLiability', 'liabilityToAsset', 'assetToEquity'
    ]
    for i in combine_columns:
        quarter_97_price_1[i] = quarter_97_2[i].to_list()
    csi300_quarter_1 = csi300_quarter.copy()
    quarter_97_price_1['csi300_close'] = csi300_quarter_1['close'].to_list()
    quarter_97_price_1.to_csv('quarter_97_combine.csv', index=False, encoding='gbk')

    # calculate price change
    quarter_97_price_1 = pd.read_csv('quarter_97_combine.csv', index_col="date", parse_dates=True, encoding='gbk')
    quarter_97_price_2 = quarter_97_price_1.copy()
    stock_change_list = []
    csi_change = []
    for i in code_list:
        df_tmp = quarter_97_price_2[quarter_97_price_2['code'].isin([i])]
        df_tmp_1 = df_tmp.copy()
        close_list = df_tmp_1['close'].to_list()
        csi_close = df_tmp_1['csi300_close'].to_list()
        for j in range(len(close_list)):
            if j == 0:
                stock_change_list.append(0)
                csi_change.append(0)
            else:
                change = (close_list[j]-close_list[j-1])/close_list[j-1]
                stock_change_list.append(change)
                change_1 = (csi_close[j]-csi_close[j-1])/csi_close[j-1]
                csi_change.append(change_1)
    # print(len(stock_change_list), len(csi_change))
    quarter_97_price_2['stock_p_change'] = stock_change_list
    quarter_97_price_2['csi300_p_change'] = csi_change
    quarter_97_price_2.to_csv('quarter_97_p_change.csv', index=False, encoding='gbk')

    # build data set，need to restructure the dataframe, relocate columns
    # data_columns = [
    #     'code', 'pubDate', 'open', 'high', 'low', 'close', 'csi300_close', 'stock_p_change', 'csi300_p_change', 'amount', 'turn', 
    #     'peTTM', 'pbMRQ', 'ma10', 'ma20', 'ma30', 'ma60', 'vma5', 'vma60', 'roeAvg', 'npMargin', 'netProfit', 'epsTTM', 'liqaShare', 'totalShare', 
    #     'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'YOYLiability', 'liabilityToAsset', 'assetToEquity'
    # ]
    # data_columns = [
    #     'code', 'pubDate', 'open', 'high', 'low', 'close', 'csi300_close', 'stock_p_change', 'csi300_p_change', 'amount', 'turn', 
    #     'ma60', 'vma60','roeAvg','netProfit','liabilityToAsset'
    # ]
    data_columns = [
        'code', 'pubDate', 'open', 'high', 'low', 'close', 'csi300_close', 'stock_p_change', 'csi300_p_change', 'amount', 'turn', 
        'ma60', 'vma60','roeAvg','netProfit','YOYEPSBasic','assetToEquity','liabilityToAsset'
    ]
    data_set = pd.DataFrame(columns=data_columns)
    quarter_97_price_3 = quarter_97_price_2.copy()
    for i in data_columns:
        column_tmp = quarter_97_price_3[i].to_list()
        data_set[i] = column_tmp
    data_set.to_csv('data_set.csv', index=False, encoding='gbk')

    # create train and test data set
    data_set_1 = data_set.copy()
    test_set = pd.DataFrame(columns=data_columns)
    train_set = pd.DataFrame(columns=data_columns)
    for i in code_list:
        df_tmp = data_set_1[data_set_1['code'].isin([i])]
        df_tmp = df_tmp.tail(1)
        df_tmp['open'] = [0]
        df_tmp['high'] = [0]
        df_tmp['low'] = [0]
        df_tmp['close'] = [0]
        df_tmp['csi300_close'] = [0]
        df_tmp['stock_p_change'] = [0]
        df_tmp['csi300_p_change'] = [0]
        test_set = test_set.append(df_tmp)
        df_tmp_1 = data_set_1[data_set_1['code'].isin([i])]
        df_tmp_1.drop(df_tmp_1.index[-1],inplace=True)
        train_set = train_set.append(df_tmp_1)
    test_set.to_csv('test_set.csv', index=False, encoding='gbk')
    train_set.to_csv('train_set.csv', index=False, encoding='gbk')

def predict(a_ticker):
    test_set = pd.read_csv('test_set.csv', encoding='gbk')
    train_set = pd.read_csv('train_set.csv', encoding='gbk')
    features = train_set.columns[9:]
    x_train = train_set[features].values
    outperformance = 0.1
    y_train = list(train_set['stock_p_change'] - train_set['csi300_p_change'] >= outperformance)

    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(x_train, y_train)
    test_set_1 = test_set.copy()
    test_features = test_set_1.columns[9:]
    x_test = test_set_1[test_features].values
    to_pred = test_set_1["code"].values
    
    y_test = clf.predict(x_test)
    to_invest = to_pred[y_test].tolist()
    result = {"good_tickers": to_invest}
    print(f"{len(to_invest)} stocks predicted to outperform the CSI 300 by more than {outperformance*100}%: " + " ".join(to_invest))
    if a_ticker in to_invest:
        print(f"Yes! the stock {a_ticker} is predicted to outperform the CSI 300 by more than {outperformance*100}%")
        result["if_your_ticker_good"]= True
    else:
        print(f"Sorry! the stock {a_ticker} is NOT predicted to outperform the CSI 300 by more than {outperformance*100}%")
        result["if_your_ticker_good"]= False
    return result

def preprocess_and_predict(a_ticker):
    df_tickers = pd.read_excel(r"Tickers_predict.xlsx")
    code_value_list = df_tickers["Ticker"].values.tolist()

    if os.path.exists('test_set.csv') and os.path.exists('train_set.csv'):
        if (a_ticker in code_value_list):
            result = predict(a_ticker)
        else:

            df_tickers.loc[len(df_tickers)] = a_ticker
            df_tickers.to_excel("Tickers_predict.xlsx", index=False, encoding='utf-8')
            preprocess()
            result = predict(a_ticker)
    else:
        if (a_ticker in code_value_list):
            preprocess()
            result = predict(a_ticker)
        else:
            df_tickers.loc[len(df_tickers)] = a_ticker
            df_tickers.to_excel("Tickers_predict.xlsx", index=False, encoding='utf-8')
            preprocess()
            result = predict(a_ticker)
    return result


# if __name__ == '__main__':
#     preprocess_and_predict("sh.600000")
