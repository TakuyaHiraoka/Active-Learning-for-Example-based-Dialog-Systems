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

Corpus for simulated environment [(downloadable from here)](https://www.dropbox.com/s/n7s8nd0x8jv4zbh/Data_OracleCorpus.zip?dl=0)

## Quick start 
1. Install all requirements. 
2. Unzip corpus for simulated environment (Data_OracleCorpus.zip), and copy corpus accoring to experiment domain. For example, if you want to peform an experiment in bus information domain, you need to copy "OracleCorpus.bu.BusInfo(DSTC1)" directory into "/Active-Learning-for-Example-based-Dialog/OracleCorpus".
3. Indicate experiment domain by setting value to "corpusType" in "/Active-Learning-for-Example-based-Dialog/ActiveConstructionofExamplBase.py". For example, if you want to perform an enperiment in bus information domain (i.e., copy "OracleCorpus.bu.BusInfo(DSTC1)" insto "OracleCorpus"), you need to set value "BusInfo": corpusType="BusInfo"
4. Indicate query selection methods by setting value to "creationMethod" in "/Active-Learning-for-Example-based-Dialog/ActiveConstructionofExamplBase.py". For example, if you want to use uncertain sampling, you need to set value "MinInExample": creationMethod="MinInExample"

## Tips
TBA

## Licence
TBA

## Author

[TakuyaHiroka](http://isw3.naist.jp/~takuya-h/)

