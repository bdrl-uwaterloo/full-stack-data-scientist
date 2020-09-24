# 优股分类
本案例结合机器学习算法中的随机森林对股票进行优劣股票分类和建模，这将为决策者提供判断优质股票的机制。本案例分析目的是让大家熟悉使用机器学习技术，因此请谨慎将其用于现实股市交易。

股票价格走势的“复杂性”与许多因素有关，例如政治稳定性，市场新闻，财务报告和股票市场法规等。 有效市场 假说的概念指出，股价反映了所有当前公开的信息。因此，其他因素导致的价格变化都是不可预测的。(当然，也有一些人持反对意见，他们认为通过某种方法和技术，对未来价格信息的获取是可能实现的)

虽然定义“好”股票都各有不同，但作为风险规避的投资者们，始终倾向于将大盘指数作为选择股票的基准。那么基于这个条件，我们如何帮助投资者选择优的股票呢？

# 问题定义：
预测本身是一项具有挑战性的研究。由于数据源和算法的可用性不断扩展，使得一些先进的技术涉及机器学习在不断被提及和使用。如今，随着互联网和计算技术的飞速发展，在股票市场上执行操作的速率已增加到几分之一秒。因此，当机立断的股票选择才能让投资者胜卷在握。在此案例研究中，将使用随机森林(Random Forest（参考ch6_ml了解更多)来为投资群判定优劣股票。

上述问题的解决方案仅需要通过股票历史数据就可以解决，根据每日价格，价格均线指标，变现率，负债率，增长率等其他公司财务报表指标将股票分为“优”和“非优”股票。换句话说，以上的指标将为模型提供预测变量（自变量）

其实模型的逻辑并不复杂，主要任务是提取和整合股票信息作为输入变量。以此预测股票季度回报是否会超越大盘指数。模型预测出来的结果将会返回True，表示股票判断为为“优”股，否则返回False 表示“非优”。

## 开发环境
 该项目使用3.6版本的Python，涉及的基本数据分析库包括 pandas，NumPy 和 Scikit-learn，数据API包括baostock和 yfinance。需要注意的是，如果使用较低版本的Python 版本（<3.6），则可能会在数据格式或使用中发现语法错误。为了避免过程中出错，我们还是建议您将python版本升级到3.6。在这里，我们总结了需要安装的数据库的基本信息，请在终端中运行以下代码进行安装：

```sh
pip install pandas==1.0.5
pip install baostock==00.8.80
pip install yfinance==0.1.54
pip install numpy==1.19.0
pip install scikit_learn==0.22
```
Baostock API经常进行更新升级，因此我们建议您升级到最新版本，如果您想升级，请在终端运行以下代码:
```sh
pip install --upgrade baostock
```

另外还有一点需要注意，根据您的IP地址，使用pip进行安装时可能会遇到“连接超时错误”的提醒。如果看到此错误消息，无需担心，使用以下方法即可：
```sh
pip install baostock -i http://mirrors.aliyun.com/pypi/simple/
```

## 项目运行
*  [数据下载] -  运行ashare_price.py 和 ashare_foundamentals.py来使用Baostoke API下载选定股票的数据。这里，我们需要用到电脑终端来执行命令。首先，需要更改目录（cd）到 stock_price_prediction文件夹中，然后再分别下载数据。具体可参以下命令：
```sh
cd stock_price_prediction
python ashare_price.py 0 1 index
python ashare_price.py 1 105 stock
python ashare_foundamentals.py 1 105 gro
python ashare_foundamentals.py 1 105 pro
python ashare_foundamentals.py 1 105 bal
```
- index 代表CSI 300 指数的价格信息，即代号为0_sh.000300
- stock 是除sh.000300以外所有股票的每日价格信息
- bal代表公司季报偿债能力指标数据
- gro代表季报成长能力指标数据
- pro代表季报盈利能力指标数据

* [获取用户界面] - 当您下载完所有数据后，您可以直接在终端执行predict_stock.py
```sh
python predict_stock.py
```
* [终止服务器运行] - 如果您想要停止服务器，只需在终端中按CTRL+C。