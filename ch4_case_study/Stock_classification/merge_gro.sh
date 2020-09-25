#!/bin/bash

echo "code,pubDate,statDate,YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI" > quarterly_gro.csv
cd downloads_gro
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../quarterly_gro.csv
done