## Name
AL4EBDS

## Overview
This code is for replicating experiments in [my IWSDS 2016 paper (to appear)](http://www.iwsds.org/)

## Description
AL4EBDS provides active learning framework for rapid development of example-based dialog system.
This framework includes query selection methods (such uncertain sampling, and information density method) for constructing example database.
For evaluating query selection methods, simulated environments based on several dialogue corpus are also provided. 

## Requirement
NLTK 3.0.2+
Sklearn 0.16.1+
Corpus for simulated environment [downloadable from here](https://www.dropbox.com/s/n7s8nd0x8jv4zbh/Data_OracleCorpus.zip?dl=0)

## Quick start 
1. Install all requirements. 
2. Unzip Data_OracleCorpus.zip, and copy the corpus accoring to experiment domain. For example, if you want to peform evaluation in bus information domain, you need to copy "OracleCorpus.bu.BusInfo(DSTC1)" into "/Active-Learning-for-Example-based-Dialog/OracleCorpus".
3. Indicate experiment domain by setting value to "corpusType" in "/Active-Learning-for-Example-based-Dialog/ActiveConstructionofExamplBase.py". For example, if you select bus information domain in previous step, you need to set value "BusInfo": corpusType="BusInfo"
4. Indicate query selection methods by setting value to "creationMethod" in "/Active-Learning-for-Example-based-Dialog/ActiveConstructionofExamplBase.py". For example, if you want to use uncertain sampling, you need to set value "MinInExample": creationMethod="MinInExample"

## Tips
TBA

## Licence
TBA

## Author

[TakuyaHiroka](http://isw3.naist.jp/~takuya-h/)



## Moomin

　　　　　　　∧　 ∧
　　　　　　 |1/　|1/
　　　　 ／￣￣￣｀ヽ、
　　　 /　　　　　　　　ヽ
　　　/　 ⌒　 ⌒　　　 |
　　　|　（●） （●）　　 |
　　 /　　　　　　　　　　|
　 /　　　　　　　　　　　|
　{　　　　　　　　　　　　|
　 ヽ、　　　　　　　ノ　　|
　　　｀`ー――‐''"　　　|
　　　 /　　　　　　　　　　|
　　　|　　　　　　　　　　|　|
　　 .|　　　　　　　　|　　|　|
　　 .|　　　　　　　　し,,ノ　|
　　　!、　　　　　　　　　　/
　　　 ヽ､　　　　　　　　 / 、
　　　　　ヽ、　 ､　　　／ヽ.ヽ、
　　　　　　 |　　|　　 |　　　ヽ.ヽ、
　　　　　 （＿_（＿＿|　　　　 ヽ､ニ三
