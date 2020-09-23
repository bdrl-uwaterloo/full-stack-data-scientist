#!/bin/bash

echo "code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare" > quarterly_pro.csv
cd downloads_pro
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../quarterly_pro.csv
done