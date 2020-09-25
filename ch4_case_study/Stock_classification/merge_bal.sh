#!/bin/bash

echo "code,pubDate,statDate,currentRatio,quickRatio,cashRatio,YOYLiability,liabilityToAsset,assetToEquity" > quarterly_bal.csv
cd downloads_bal
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../quarterly_bal.csv
done