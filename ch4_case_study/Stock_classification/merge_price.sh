#!/bin/bash

echo "date,code,open,high,low,close,volume,amount,adjustflag,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,pctChg,isST" > daily_price.csv
cd downloads_price
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../daily_price.csv
done