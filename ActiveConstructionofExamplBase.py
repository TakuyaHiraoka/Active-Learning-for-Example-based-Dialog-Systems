# coding:utf-8
#!/usr/bin/env python
'''
Created on 2015/01/30

@author: takuya-hv2
'''
import glob
import os
import re
import codecs
import sys,datetime
import copy
import os
from sklearn.feature_extraction.text import *
import codecs
from sklearn.metrics.pairwise import *
import pickle
import unicodedata
from sklearn.externals import joblib
from numpy import random
import math
import json
import unicodedata
import string
import nltk
import cPickle as pickle
import gzip
import zlib
#能動学習のシュミレーション評価の枠組み
#V1からの修正点：
#-1)初期用例データベースをランダムに初期化(合わせて一部のデータ構造の変更)
#-2)ホールドアウトしたテストセットに対するCosine類似度を評価項目とした
#-3)1回のステップごとにk個のアノテーションをエキスパートに指示可能にした
#V2からの修正点：
#-1)バグフィックス：BaselineとProposedで能動学習がとまるステップサイズが異なる点を修正
#-2)Proposedとして：　クエリ選択の基準として、用例中の発話との類似度の最大値が小ささを用いる
#現在の実験： subset_a_AKT_Aの100個のコーパスをオラクルとして利用して実験
#V4の特徴：
#中規模の評価用データ（AKT１００対話）を用いた評価に利用
#V4からの修正点：
#-1)コサイン類似度と提案手法の性能の相関分析.それにともなってRandom手法の削除
#V5の特徴
#V6の特賞
#iroiroなOracleを利用
#V7の特徴-クエリに含まれる入力なるべくばらけさせる
#V8からの変更点：用例中の最大類似度とOracle中の平均類似度の幾何平均による選択
#V9からの変更点
#1.クエリ選択部のクラス化。実験条件クラスの導入
#2.計算の高速化（クエリ選択時の評価値のキャッシュ、システム評価時の評価値のキャッシュ、ユーザ発話プール内の計算の打ち切り上限の導入）
#3.クエリ選択手法の追加(SimPoolのみ、SimとSimPoolの算術平均)
#4.性能評価の際、複数の同一評価値の用例が存在する際はランダムに選択
#5.評価値を反転させる実験条件も追加
#V11からの変更点
#1.Switchboardコーパスを用いた実験に対応
#-.
#V12からの変更点
#1.既にクエリに追加したものを除外する変数の考慮
#2.時間経過によるスコアの減衰を考慮して算術平均をとる手法の導入
#V14からの変更点
#1.スコアを計算するコーパスサイズに基づいた重みを用いたArith, Geoの導入
#V15からの変更点
#1.Idoの映画コーパスの読み込みを導入(Turn2->Turn3)のみ利用
#V16からの変更点
#1. tf-idf重みを追加
#2. orijinal_sentence->actual_sentenceに変更（固有名詞の置換やパラ言語情報の削除）
#3.Ido's corpusのテキストに対して標準化を導入
#4.英語のベクトル化の際に、正規化を導入　http://d.hatena.ne.jp/torasenriwohashiru/20110806/1312558290
#-1.英語の文章に関してステミング（語幹情報のみを利用）を導入
#-2.英語の文章に関してレンマ化（見出し語化）を導入 
#V17からの変更点
#1.平均計算の方法の修正<----結局バグは無かったので、修正は保留
#2.テストデータを一様に選択（シードを固定してランダムにサンプリング）
#V19からの変更点
#1.不要なCWGeoの削除
#2.システムの発話の推定値を用いた手法,SysMinInExampleの導入
#V20からの変更点
#1.各ステップで上手区答えられなかった応答スコア（類似度）を出力
#V21からの変更点
#1.DSTCで使われたCMU Buns informationドメインのテストデータを導入

#V22からの変更点
#1.DSTC4で使われたGuideドメインのテストデータを導入

#V23からの変更点
#1.TRAINSドメインのテストデータを導入

#V24からの変更点
#1.Cleverbotドメインのテストデータを導入

#V25からの変更点
#Restrant, Tourist domainのテストデータの導入

#V26からの変更点
#評価スコアに基づいて確率的にクエリを選択

#V28からの修正点
#GeoMinAndAvrMaxに確率的にクエリを選択する手法を導入

#実験条件設定用変数
class ExperimentalCondition:
    #手法に関する実験変数
    creationMethod="GeoMinAndAvrMax"#
    #Random:ランダムに作成追加
    #MinInExample:用例中の最大類似度が低いユーザ発話に対する用例を優先的に作成・追加
    #AvrMaxInPool:ユーザ発話のプール中の発話間で平均類似度が高いものをユーザ発話に対する用例を優先的に作成・追加
    #ArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均
    #GeoMinAndAvrMax: MinInExampleとAvrMaxInPoolの相乗平均
    #DWArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均。ただし、重みが時間によって変化
    #CWArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均。ただし、スコアを計算するコーパスサイズに基づいて重みが変化
    #ESysMinSimInExample: 
    #ESysAvrMaxInPool:
    isInverseScore=False#評価値を順序を反転させる。ランダム手法以外の手法に適用
    isIgnoreOverlappedQuery=True#これまでのクエリ中に同じものが含まれるものは除外する#全手法に適用
    weightDecayOfAvrMaxInPoolAtEachStep=0.9#各ターンにおけるAvrMaxInPoolの重みの減衰値(1.0/この値)
    isInverseWeight=True#CWArithMinAndAvrMaの重みを反転させる
    #
    isSamplingQueriesBasedOnScore=True#V28で追加　確率的にクエリサンプリングするか（MinInExample, AvrMaxInPoolのみに導入）
    
    #対象データに対する実験変数
    corpusType="TouristInfo"
    #ProjectNextNLP
    #Switchboard
    #IdosMovie
    #BusInfo
    #GuideDomain
    #Trains
    #Cleverbot
    #RestrantInfo
    #TouristInfo
    
    isInitialyCreateCompiledUserSystemUteranceAndVecorizor=False#実験用データをファイルから構築しなおすか
    isUseTfIDFweight=True#実験用データをファイルから構築しなおす際に、tf-idf重み付けを利用するか
    num4HoldoutTestSet=200#8000#8000#9500#1200#2000#7000#8500#15000#8500#6500#1000#30#全オラクルデータ中の、テストセット数
    percentile4InitialExampleDatabaseSize=0.005#0.002#0.0021#0.005#0.001#0.005#0.01#全オラクルデータ中の、初期のシステムの用例が占める割合
    numberOfMaxSystem=50#初期化する回数
    numberOfMaxStep=100#60#50#能動学習を繰り返すステップ数
    numberofQueryAtEachTurn=100#100#20#各ターンでエキスパートにシステムの応答数作成をいらいする数
    
    #プール内計算近似用の変数＠creatExamplePairs_AvrMaxInPool
    maxNumberOfUserUtteranceForQueryCandidate=10000#15000#10000#7000#プール中のユーザ発話の内先頭から何個データを利用するか
    maxNumberOfUserUtteranceAsOpponent=100#プール中のユーザ発話間の平均類似度を計算するためにサンプルするデータ数
    #システムの発話推定計算近似用の変数@
    maxNumberOfSamlingForSystemUtteranceEstimatation=100#システムの発話を推定するためにサンプルする回数
    
    #クエリのトレース
    isTraceQuery=True
    
    
    
#用例の追加(creatExamplePairs)を行うクラス
class ExamplePairsCreator:
    #関数CreatExamplePairs_~
    #入力として、exampleDataBase, remainingOracle
    #出力としてreturn createdExamplePairs, remainingOracleを返す
    #examlpleBase: 現在のシステム用例、 
    #remainingOracle: ユーザの発話用例のプール
    #createdExamplePairs:　作成された用例
    
    #高速化のキャッシング用クラス変数
    similarityUserUtteranceInExampleAndInPool={}#[プール中のユーザ発話][用例中のユーザ発話]＝類似度
    similarityUserUtteranceInPool={}#[プール中のユーザ発話][用例中のユーザ発話]＝類似度
    #高速化のキャッシング用インスタンス変数
    #-creatExamplePairs_MinSimInExample計算用
    bestSimAndUserUtteranceInExampleTowardPool=None#[プール中のユーザ発話]=[もっとも類似している用例中のユーザ発話のスコア,もっとも類似している用例中のユーザ発話]
    createdPairInMostPrevious=None#直前に作成された用例群
    #既に過去のクエリで追加したもの＠オーバーラップする変数は取り除くよう
    dicAlreadyQueried=None
    #DWArithMinAndAvrMax用
    weightOfAvrMaxInPool=None
    #ESysAvrMaxInPool
    staticSortedID=None
    
    def __init__(self):
        self.bestSimAndUserUtteranceInExampleTowardPool={}
        self.createdPairInMostPrevious=None
        #
        self.dicAlreadyQueried={}
        #
        self.weightOfAvrMaxInPool=1.0
        #ESysAvrMaxInPool用
        self.dynamicSortedID=None
        
    
    def creatExamplePairs_random(self,exampleDataBase, remainingOracle):
        #プールのユーザ発話をランダムに選択して、用例の追加を行う
        createdExample=[]
        numQuery=0
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(remainingOracle)>0):
                query=remainingOracle.pop(random.randint(low=0,high=99999)%len(remainingOracle))
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
            #print str(len(createdExample))
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(remainingOracle)>0):
                createdExample.append(remainingOracle.pop(random.randint(low=0,high=99999)%len(remainingOracle)))
                numQuery+=1
        return createdExample, remainingOracle


    def creatExamplePairs_MinSimInExample(self,exampleDataBase, remainingOracle):
        #用例中の全発話との最小コサイン類似度が最も少ない発話を、エキスパートシミュレータが作成しして追加
        count=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                dicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                dicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1

        #ソーティング
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()
        sortedID=[]
        sortedScore=[]#Appended part in V28
        for elem in v:
            sortedID.append(elem[0])
            sortedScore.append(elem[1])#Appended part in V28
        ind=0
        averageMaxSimInQuery=0.0
        cumScore=0.0#Appended part in V28
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            cumScore+=(1.0-elem[1])#Appended part in V28
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0

        
        #Appended part in V29:----------------------------------
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=(1.0-sortedScore[ind])
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=(1.0-sortedScore[ind])
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    if query[0] not in self.dicAlreadyQueried:
                        self.dicAlreadyQueried[query[0]]=1
                        createdExample.append(query)
                        numQuery+=1
                    else:
                        self.dicAlreadyQueried[query[0]]+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=(1.0-sortedScore[ind])
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=(1.0-sortedScore[ind])
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    createdExample.append(query)
                    numQuery+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        #Appended part in V29:----------------------------------

        
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle
    
    
    def creatExamplePairs_AvrMaxInPool(self,exampleDataBase, remainingOracle):
        #AvrMaxInPool:ユーザ発話のプール中の発話間で平均類似度が高いものをユーザ発話に対する用例を優先的に作成・追加
        ind=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        opponentIDs=[]
        for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
            opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
        for oracle in remainingOracle:
            avrSim=0.0
            if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                break
            for opponentID in opponentIDs:
                oOracle=remainingOracle[opponentID]
                if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                else:
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    
                    avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
            avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
            dicUserInSim[ind]=(1.0-avrSim)
            ind+=1
            
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()        
        sortedID=[]
        sortedScore=[]#Appended part in V28
        for elem in v:
            sortedID.append(elem[0])
            sortedScore.append(elem[1])#Appended part in V28
        ind=0
        averageMaxSimInQuery=0.0
        cumScore=0.0#Appended part in V28
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            cumScore+=(1.0-elem[1])#Modified part in V29:----------------------------------
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        surfix=0
        
        #Appended part in V29:----------------------------------
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=(1.0-sortedScore[ind])
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=(1.0-sortedScore[ind])
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    if query[0] not in self.dicAlreadyQueried:
                        self.dicAlreadyQueried[query[0]]=1
                        createdExample.append(query)
                        numQuery+=1
                    else:
                        self.dicAlreadyQueried[query[0]]+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=(1.0-sortedScore[ind])
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=(1.0-sortedScore[ind])
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    createdExample.append(query)
                    numQuery+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        #Appended part in V29:----------------------------------

                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle


    def creatExamplePairs_ArithmeticMinSimInExampleAndAvrMaxSimInPool(self,exampleDataBase, remainingOracle):
        #ArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均
        count=0
        tdicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        
        #Pooll
        ind=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        opponentIDs=[]
        for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
            opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
        for oracle in remainingOracle:
            avrSim=0.0
            if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                break
            for opponentID in opponentIDs:
                oOracle=remainingOracle[opponentID]
                if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                else:
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    
                    avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
            avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
            dicUserInSim[ind]=((1.0-avrSim)+(tdicUserInSim[ind]))/2.0
            ind+=1
            
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()
        sortedID=[]
        for elem in v:
            sortedID.append(elem[0])
        ind=0
        averageMaxSimInQuery=0.0
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        surfix=0
        
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                query=remainingOracle.pop(id)
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                createdExample.append(remainingOracle.pop(id))
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
                numQuery+=1
                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle


    
    def creatExamplePairs_GeometrixMinSimInExampleAndAvrMaxSimInPool(self,exampleDataBase, remainingOracle):
        #GeoMinAndAvrMax: MinInExampleとAvrMaxInPoolの相乗平均
        count=0
        tdicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        
        #Pooll
        ind=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        opponentIDs=[]
        for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
            opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
        for oracle in remainingOracle:
            avrSim=0.0
            if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                break
            for opponentID in opponentIDs:
                oOracle=remainingOracle[opponentID]
                if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                else:
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    
                    avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
            avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
            #dicUserInSim[ind]=(1.0-avrSim)*(tdicUserInSim[ind]+0.000001)
            dicUserInSim[ind]=(avrSim)*(1.0-tdicUserInSim[ind])##Modified part in V30:----------------------------------

            #dicUserInSim[ind]=math.sqrt(dicUserInSim[ind])#エラーが出るため削除
            ind+=1
            
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()        
        sortedID=[]
        sortedScore=[]#Appended part in V28
        for elem in v:
            sortedID.append(elem[0])
            sortedScore.append(elem[1])#Appended part in V28
        ind=0
        averageMaxSimInQuery=0.0
        cumScore=0.0#Appended part in V28
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            cumScore+=elem[1]#Modified part in V30:----------------------------------
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        surfix=0
        
        #Appended part in V29:----------------------------------
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=sortedScore[ind]
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=sortedScore[ind]
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    if query[0] not in self.dicAlreadyQueried:
                        self.dicAlreadyQueried[query[0]]=1
                        createdExample.append(query)
                        numQuery+=1
                    else:
                        self.dicAlreadyQueried[query[0]]+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
                id=None#Appended part in V28:----------------------------------
                if not ExperimentalCondition.isSamplingQueriesBasedOnScore:
                    id=sortedID.pop(0)
                else:
                    point=random.uniform(low=0.0,high=cumScore)
                    currentCumScore=0.0
                    for ind in range(len(sortedID)):
                        currentCumScore+=sortedScore[ind]
                        if point <= currentCumScore:
                            id=sortedID[ind]
                            id=int(id)
                            cumScore-=sortedScore[ind]
                            #print sortedScore[ind]
                            #print sortedScore
                            #print id
                            sortedID.pop(ind)
                            sortedScore.pop(ind)
                            break
                    if (id == None):
                        #print "Illigal id (NULL) 0 is interpolated"
                        id=0
                        if len(sortedID)>0:#Appended part in V29
                            id=sortedID.pop()#Appended part in V29
                    #Appended part in V28:----------------------------------
                #Appended part in V29:----------------------------------
                if (id < len(remainingOracle)) and (id >= 0):
                    query=remainingOracle.pop(id)
                    createdExample.append(query)
                    numQuery+=1
                    for l in range(len(sortedID)):
                        if sortedID[l] >= id:
                            sortedID[l]-=1
                else:
                    print "Illigal id:" + str(id)
                    print len(remainingOracle)
                #Appended part in V29:----------------------------------
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        #Appended part in V29:----------------------------------
                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle
    
    

    def creatExamplePairs_DWArithmeticMinSimInExampleAndAvrMaxSimInPool(self,exampleDataBase, remainingOracle):
        #DWArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均。ただし、重みが時間によって変化
        count=0
        tdicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        
        #Poll
        #print str(self.weightOfAvrMaxInPool)
        ind=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        opponentIDs=[]
        for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
            opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
        for oracle in remainingOracle:
            avrSim=0.0
            if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                break
            for opponentID in opponentIDs:
                oOracle=remainingOracle[opponentID]
                if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                else:
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    
                    avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
            avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
            dicUserInSim[ind]=(self.weightOfAvrMaxInPool*(1.0-avrSim)+(tdicUserInSim[ind]))/2.0
            ind+=1
            
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()
        sortedID=[]
        for elem in v:
            sortedID.append(elem[0])
        ind=0
        averageMaxSimInQuery=0.0
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        surfix=0
        
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                query=remainingOracle.pop(id)
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                createdExample.append(remainingOracle.pop(id))
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
                numQuery+=1
        #減衰
        self.weightOfAvrMaxInPool/=ExperimentalCondition.weightDecayOfAvrMaxInPoolAtEachStep
        
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle



    def creatExamplePairs_CWArithMinAndAvrMax(self,exampleDataBase, remainingOracle):
        #CWArithMinAndAvrMax: MinInExampleとAvrMaxInPoolの加算平均。ただし、スコアを計算するコーパスサイズに基づいて重みが変化
        sumZ=float(len(remainingOracle))+float(len(exampleDataBase))
        weight4MinInExample=float(len(exampleDataBase))
        weight4AvrMaxInPool=float(len(remainingOracle))
        #print "CWArith"
        #print sumZ
        #print weight4MinInExample
        #print weight4AvrMaxInPool
        weight4MinInExample/=sumZ
        weight4AvrMaxInPool/=sumZ
        #print weight4MinInExample
        #print weight4AvrMaxInPool

        count=0       
        tdicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]                
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                tdicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        
        #Pooll
        ind=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        opponentIDs=[]
        for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
            opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
        for oracle in remainingOracle:
            avrSim=0.0
            if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                break
            for opponentID in opponentIDs:
                oOracle=remainingOracle[opponentID]
                if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                    ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                else:
                    if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(oracle[2], oOracle[2])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    
                    avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
            avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
            if not ExperimentalCondition.isInverseWeight:
                dicUserInSim[ind]=( ((1.0-weight4AvrMaxInPool)*(1.0-avrSim)) + ((1.0-weight4MinInExample)*(tdicUserInSim[ind])) )/2.0
            else:
                dicUserInSim[ind]=( ((weight4AvrMaxInPool)*(1.0-avrSim)) + ((weight4MinInExample)*(tdicUserInSim[ind])) )/2.0
                
            ind+=1
            
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()
        sortedID=[]
        for elem in v:
            sortedID.append(elem[0])
        ind=0
        averageMaxSimInQuery=0.0
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        surfix=0
        
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                query=remainingOracle.pop(id)
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                createdExample.append(remainingOracle.pop(id))
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
                numQuery+=1
                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle
    
    
    
    #
    def creatExamplePairs_ESysMinSimInExample(self,exampleDataBase, remainingOracle):
        #用例中のシステムの全発話との（推定したシステム発話の）最小コサイン類似度が最も少ない発話を、エキスパートシミュレータが作成しして追加
        count=0
        dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
        if self.createdPairInMostPrevious == None:
            #一番最初のステップの場合のみ全通り計算
            for oracle in remainingOracle:
                maxSim=-0.1
                #サンプリングによるAの推定値
                estimatedSysU=None
                cum=0.0
                for i in range(ExperimentalCondition.maxNumberOfSamlingForSystemUtteranceEstimatation):
                    sampleSysU=exampleDataBase[random.randint(low=0,high=99999)%len(exampleDataBase)]
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    else:
                        if sampleSysU[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]
                    sim+=0.000001#ベクトルが0になるのを防ぐため
                    if estimatedSysU == None:
                        estimatedSysU=sim*sampleSysU[3]
                        cum+=sim
                    else:
                        estimatedSysU=np.add(estimatedSysU,sim*sampleSysU[3])
                        cum+=sim
                estimatedSysU=estimatedSysU/cum
                
                #推定したしたシステム応答に基づいた用例の比較
                for examplePair in exampleDataBase:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(estimatedSysU, examplePair[3])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(estimatedSysU, examplePair[3])[0][0]
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if maxSim < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                        maxSim=sim
                    
                dicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1
        else:#2ステップ目以降は差分のみ計算
            for oracle in remainingOracle:
                #サンプリングによるAの推定値
                estimatedSysU=None
                cum=0.0
                for i in range(ExperimentalCondition.maxNumberOfSamlingForSystemUtteranceEstimatation):
                    sampleSysU=exampleDataBase[random.randint(low=0,high=99999)%len(exampleDataBase)]
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    else:
                        if sampleSysU[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]
                    sim+=0.000001#ベクトルが0になるのを防ぐため
                    if estimatedSysU == None:
                        estimatedSysU=sim*sampleSysU[3]
                        cum+=sim
                    else:
                        estimatedSysU=np.add(estimatedSysU,sim*sampleSysU[3])
                        cum+=sim
                estimatedSysU=estimatedSysU/cum

                for examplePair in self.createdPairInMostPrevious:
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(estimatedSysU, examplePair[3])[0][0]
                    else:
                        if examplePair[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]=cosine_similarity(estimatedSysU, examplePair[3])[0][0]
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][examplePair[0]]
                    
                    if self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0] < sim:
                        self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]]=[sim,examplePair[0]]
                
                dicUserInSim[count]=self.bestSimAndUserUtteranceInExampleTowardPool[oracle[0]][0]
                count+=1

        #ソーティング
        v = sorted(dicUserInSim.items(), key=lambda x:x[1])
        if ExperimentalCondition.isInverseScore:
            v.reverse()
        sortedID=[]
        for elem in v:
            sortedID.append(elem[0])
        ind=0
        averageMaxSimInQuery=0.0
        for elem in v:
            averageMaxSimInQuery+=elem[1]
            ind+=1
        averageMaxSimInQuery /= float(len(v))
        print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
        f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")
        numQuery=0
        
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                query=remainingOracle.pop(id)
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
            #print len(sortedID)
            #print str(numQuery)
            #print str(len(createdExample))
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(sortedID)>0):
                id=sortedID.pop(0)
                createdExample.append(remainingOracle.pop(id))
                for l in range(len(sortedID)):
                    if sortedID[l] >= id:
                        sortedID[l]-=1
                numQuery+=1
                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle



    def creatExamplePairs_ESysAvrMaxInPool(self,exampleDataBase, remainingOracle):
        #AvrMaxInPool:ユーザ発話のプール中の発話間で平均類似度が高いものをユーザ発話に対する用例を優先的に作成・追加
        if ExamplePairsCreator.staticSortedID == None:
            #一番最初のステップの場合のみ全通り計算
            ind=0
            dicUserInSim={}#能動学習用[ユーザの入力ID]=類似度
            opponentIDs=[]
            estimatedUtt={}
            #サンプリングによる質問に対するAの推定値の計算
            for oracle in remainingOracle:
                maxSim=-0.1
                estimatedSysU=None
                cum=0.0
                for i in range(ExperimentalCondition.maxNumberOfSamlingForSystemUtteranceEstimatation):
                    sampleSysU=exampleDataBase[random.randint(low=0,high=99999)%len(exampleDataBase)]
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool:
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    else:
                        if sampleSysU[0] not in ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]=cosine_similarity(oracle[2], sampleSysU[2])[0][0]
                    sim=ExamplePairsCreator.similarityUserUtteranceInExampleAndInPool[oracle[0]][sampleSysU[0]]
                    sim+=0.000001#ベクトルが0になるのを防ぐため
                    if estimatedSysU == None:
                        estimatedSysU=sim*sampleSysU[3]
                        cum+=sim
                    else:
                        estimatedSysU=np.add(estimatedSysU,sim*sampleSysU[3])
                        cum+=sim
                estimatedSysU=estimatedSysU/cum
                estimatedUtt[oracle[0]]=estimatedSysU
            #Aの推定値を用いた計算
            for i in range(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent):
                opponentIDs.append(random.randint(low=0,high=99999999)%len(remainingOracle))
            for oracle in remainingOracle:
                avrSim=0.0
                if ind >= ExperimentalCondition.maxNumberOfUserUtteranceForQueryCandidate:
                    break
                for opponentID in opponentIDs:
                    oOracle=remainingOracle[opponentID]
                    if oracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(estimatedUtt[oracle[0]], estimatedUtt[oOracle[0]])[0][0]
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                        ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                    else:
                        if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]]:
                            ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]=cosine_similarity(estimatedUtt[oracle[0]], estimatedUtt[oOracle[0]])[0][0]
                            if oOracle[0] not in ExamplePairsCreator.similarityUserUtteranceInPool:
                                ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]]={}
                            ExamplePairsCreator.similarityUserUtteranceInPool[oOracle[0]][oracle[0]]=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                        
                        avrSim+=ExamplePairsCreator.similarityUserUtteranceInPool[oracle[0]][oOracle[0]]
                avrSim/=(float(ExperimentalCondition.maxNumberOfUserUtteranceAsOpponent))
                dicUserInSim[ind]=(1.0-avrSim)
                ind+=1
                
            v = sorted(dicUserInSim.items(), key=lambda x:x[1])
            if ExperimentalCondition.isInverseScore:
                v.reverse()        
            ExamplePairsCreator.staticSortedID=[]
            for elem in v:
                ExamplePairsCreator.staticSortedID.append(elem[0])
            ind=0
            averageMaxSimInQuery=0.0
            for elem in v:
                averageMaxSimInQuery+=elem[1]
                ind+=1
            averageMaxSimInQuery /= float(len(v))
            print str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)
            f.write(str(numStep)+" turn, Average MaxSim+(1-Average Sim in RemainingOracle) of remaining database=" + str(averageMaxSimInQuery)+"\n")

        numQuery=0
        if self.dynamicSortedID == None:#各ステップの始まりにロード
            self.dynamicSortedID=copy.deepcopy(ExamplePairsCreator.staticSortedID)
        
        createdExample=[]
        if ExperimentalCondition.isIgnoreOverlappedQuery:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(self.dynamicSortedID)>0):
                id=self.dynamicSortedID.pop(0)
                query=remainingOracle.pop(id)
                if query[0] not in self.dicAlreadyQueried:
                    self.dicAlreadyQueried[query[0]]=1
                    createdExample.append(query)
                    numQuery+=1
                else:
                    self.dicAlreadyQueried[query[0]]+=1
                for l in range(len(self.dynamicSortedID)):
                    if self.dynamicSortedID[l] >= id:
                        self.dynamicSortedID[l]-=1
        else:
            while (numQuery < ExperimentalCondition.numberofQueryAtEachTurn) and (len(self.dynamicSortedID)>0):
                id=self.dynamicSortedID.pop(0)
                createdExample.append(remainingOracle.pop(id))
                for l in range(len(self.dynamicSortedID)):
                    if self.dynamicSortedID[l] >= id:
                        self.dynamicSortedID[l]-=1
                numQuery+=1
                
        self.createdPairInMostPrevious=createdExample
        return createdExample, remainingOracle



class Word2VecCompiler:
    numIteration=0
    lmtzr=nltk.WordNetLemmatizer()#Note クラスタで動かすときにはコメントアウトすること
    stm=nltk.PorterStemmer()#Note クラスタで動かすときにはコメントアウトすること
    
    #dialoguesベクト化対象の全発話のリスト
    def ConstructCimilarityCalculatorAndTfIDFVectors(self,dialogues):
        self.numIteration=0
        #vecotorizerTfidf=TfidfVectorizer(analyzer=self.stems,stop_words=[],ngram_range=(1,3))#dfでストップワードを作成しない
        vecotorizerTfidf=None
        matTfidf=None
        if ExperimentalCondition.isUseTfIDFweight:
            "#Vectorize with tf-idf"
        if ExperimentalCondition.corpusType=="ProjectNextNLP":#V12で修正
            if ExperimentalCondition.isUseTfIDFweight:
                vecotorizerTfidf=TfidfVectorizer(analyzer=self.stems,stop_words=[],ngram_range=(1,3))#カウント特徴量
            else:
                vecotorizerTfidf=CountVectorizer(analyzer=self.stems,stop_words=[],ngram_range=(1,3))#カウント特徴量
        else:#日本語以外のコーパスの場合
            if ExperimentalCondition.isUseTfIDFweight:
                vecotorizerTfidf=TfidfVectorizer(analyzer=self.split_EnglishSentence,stop_words=[],ngram_range=(1,3))#,max_df=1.0, min_df=2)#２個より出てないものはカット
            else:
                vecotorizerTfidf=CountVectorizer(analyzer=self.split_EnglishSentence,stop_words=[],ngram_range=(1,3),max_df=1.0, min_df=2)#２個より出てないものはカット
        matTfidf=vecotorizerTfidf.fit_transform(dialogues)
        
        return vecotorizerTfidf, matTfidf
        
    def _split_to_words(self,text, to_stem=False):
        self.numIteration+=1
        uText=text
        uText
        print u"Num"+str(self.numIteration)+":"+unicode(uText,"shift-jis","ignore")
        #類似文検索テスト
        testInput=text+""
        testInput=re.sub("<.+?>","",testInput)
        testInput=re.sub("\n","",testInput)
        #-分かち書き＋ＰＯＳタギング
        tempFile=open("temp4WS","w")
        tempFile.write(testInput)
        tempFile.close()
        os.system("mecab " + "temp4WS > temp4C")
        #-BOW形式に生計
        tempFile=codecs.open("temp4C","r","shift_jis")
        ResultOfMecab=u""
        for line in tempFile:
            ResultOfMecab+=line
        #print ResultOfMecab
        ResultOfMecab+=u""
        #ThirdImpelementation 全角と半角と数値の標準化
        ResultOfMecab=unicodedata.normalize('NFKC', ResultOfMecab)
        
        info_of_words = ResultOfMecab.split(u'\n')
        words = []
        for info in info_of_words:
            #ThirdImpelementation で追加　ストップ語である機能後（助詞、助動詞、接続詞）の削除　http://kotoba.nuee.nagoya-u.ac.jp/tsutsuji/
            m=re.search(u"((助詞)|(助動詞)|(接続詞))",info)
            #m=re.search(u"((助動詞)）)",info)
            if m !=None:
                #print m.group(0)
                continue
            
            # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
            if info == 'EOS' or info == 'EOS\r' or info == '':
                break
                # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
            info_elems = info.split(u',')
            # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
            if info_elems[6] == '*':
                # info_elems[0] => 'ヴァンロッサム\t名詞'
                words.append(info_elems[0][:-3])
                continue
            if to_stem:
                # 語幹に変換
                words.append(info_elems[6])
                continue
            # 語をそのまま
            words.append(info_elems[0][:-3])
        return words
    
    def words(self,text):
        words = self._split_to_words(text=text, to_stem=False)
        return words
    
    def stems(self,text):
        stems = self._split_to_words(text=text, to_stem=True)
        return stems
    
    
    #English Segmentor
    def split_EnglishSentence(self,text):
        self.numIteration+=1
        print u"Num"+str(self.numIteration)+":"+text
        #return re.split(" +", text)
        splittedSentence=re.split(" +", text)
        #print splittedSentence
        #ステミング
        i=0
        for word in splittedSentence:
            newWord=Word2VecCompiler.stm.stem(word)
            splittedSentence[i]=newWord
            i+=1
        #レンマタイズ
        i=0
        for word in splittedSentence:
            newWord=Word2VecCompiler.lmtzr.lemmatize(word)
            splittedSentence[i]=newWord
            i+=1
        #print splittedSentence
        return splittedSentence

    


if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
    #初期化 （オラクルのコーパス＋初期用例）
    #-オラクルの対話コーパスの読み込み[D..], D=[P..],P=<user utterance,system utterance, Compiled user utter,Compiled sys utterance>
    #--各ディレクトリに含まれるデータを読み込み[Dp..],Dp=[P...], P=[user utterance, sysutterance,破綻ラベル]
    dialogWithCompileInfo=[]#☆[D..], D=[P..],P=<user utterance,system utterance, Compiled user utter,Compiled sys utterance>
    compiledUserInput=None#☆ユーザのインプットリスト
    compiledSystemresponce=None#☆ユーザのインプットに対応するシステムの応答
    vectorizorForUserUtterane=None#類似度比較器
    vectorizorForSystemUtterance=None#類似度比較器
    w2vComp=Word2VecCompiler()
    if ExperimentalCondition.corpusType=="ProjectNextNLP":#V12で修正
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(ProjectNextNLP)"
            allDialogFiles=[]#全アノテーション済みの対話が格納されたファイル
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u"\.tsv$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            print "All number of conversations is " + str(len(allDialogFiles))
            
            #---Project Next NLPデータの変換 形式：<システム発話\nユーザ発話>+, システム発話=<破綻（O=破綻していない, T=破綻じゃないが違和感を感じる,X=破綻）\t非文( =なし、*=ひぶん)\t本文\tコメント\n >, ユーザ発話=<本文>
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile,u"r",u"utf-8")
                rawIOLPairs=[]
                i=1
                tempIOLPairs=[u""]
                for elem in f:
                    #ユニコードに対応していない文字の削除
                    elem=elem.replace(u"～","")
                    elem=elem.replace(u"…","")
                    elem=elem.replace(u"♡","")
                    elem=elem.replace(u"－","")
                    try: 
                        elem.encode("shift-jis")
                    except: 
                        print "Err"
                        print dialogFile
                        print elem
                    elem=elem.split(u"\t")
                    if (i%2)==1:#システム発話の場合
                        tempIOLPairs.append(elem[2][7:])
                        tempIOLPairs.append(elem[0])
                        rawIOLPairs.append(tempIOLPairs)
                        tempIOLPairs=[]
                    elif (i%2)==0:#ユーザ発話の場合
                        tempIOLPairs.append(elem[2][7:])
                    i+=1
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][2]==u'O' and (rawDialog[i][j][0] != ""):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    rawDialogWithoutIncorrectResponse[i][j].pop(2)#Only Project Next NLP needs
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()
            
            
            
            
            
    elif ExperimentalCondition.corpusType == "Switchboard":#------------------------------
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(Switchboard)"
            allDialogFilesOfA=[]#話者Aの発話を集めたもの
            allDialogFilesOfB=[]#話者Bの発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u"A-ms98-a-trans.text$", file) != None:
                        allDialogFilesOfA.append(file)
            print allDialogFilesOfA
            
            #---Project Next NLPデータの変換 形式：<システム発話\nユーザ発話>+, システム発話=<破綻（O=破綻していない, T=破綻じゃないが違和感を感じる,X=破綻）\t非文( =なし、*=ひぶん)\t本文\tコメント\n >, ユーザ発話=<本文>
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            for dialogFile in allDialogFilesOfA:
                f=codecs.open(dialogFile,u"r",u"utf-8")
                fb=codecs.open(dialogFile.replace("A-ms98-a-trans","B-ms98-a-trans"),u"r",u"utf-8")
                rawIOLPairs=[]
                tempIOLPairs=[u""]
                
                #Switchboardの各話者の発話データをそれぞれ読み込む
                #この際、[Silence]のみの発話は無視（読みこまない）
                AInfos=[]
                for elem in f:
                    try: 
                        elem.encode("shift-jis")
                    except: 
                        print "Err"
                        print dialogFile
                        print elem
                    if re.search(u"\[silence\]",elem) == None:
                        AInfos.append(re.split(u" +",elem))
                BInfos=[]
                for elem in fb:
                    try: 
                        elem.encode("shift-jis")
                    except: 
                        print "Err"
                        print dialogFile
                        print elem
                    if re.search(u"\[silence\]",elem) == None:
                        BInfos.append(re.split(u" +",elem))
                
                #パラ言語情を削除したAとBの発話をそれぞれユーザ、システムの発話とする
                #開始時間早い順に発話が読み込まれユーザ発話->システム発話とする?
                UserUtt=""
                SysUtt=""
                switchCount=0
                prevSpeaker=None
                currentSpeaker=""
                while (len(AInfos) > 0) and (len(BInfos) > 0):
                    prevSpeaker=currentSpeaker
                    if (len(AInfos) > 0) and (len(BInfos) > 0):
                        if float(AInfos[0][1])<float(BInfos[0][1]):
                            for w in AInfos.pop(0)[3:]:
                                w=re.sub("\n","",w)
                                w=re.sub("\[.*\]","",w)                                
                                UserUtt+=" "+w
                            currentSpeaker="A"
                        else:
                            for w in BInfos.pop(0)[3:]:
                                w=re.sub("\n","",w)
                                w=re.sub("\[.*\]","",w)                                
                                SysUtt+=" "+w
                            currentSpeaker="B"
                        if currentSpeaker != prevSpeaker:
                            switchCount+=1
                        if switchCount >=2:
                            #print UserUtt + "->" + SysUtt
                            rawIOLPairs.append([UserUtt,SysUtt])
                            UserUtt=""
                            SysUtt=""
                            switchCount=0
                           
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()



    elif ExperimentalCondition.corpusType == "IdosMovie":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(IdosMovies)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*txt$", file) != None:
                        allDialogFiles.append(file)
            print allDialogFiles
            
            #---Project Next NLPデータの変換 形式：<システム発話\nユーザ発話>+, システム発話=<破綻（O=破綻していない, T=破綻じゃないが違和感を感じる,X=破綻）\t非文( =なし、*=ひぶん)\t本文\tコメント\n >, ユーザ発話=<本文>
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile,"r")#,u"r",u"utf-8")
                jsonData=[]
                for s in f:
                    jsonData.append(json.loads(s))
                    #print jsonData
                rawIOLPairs=[]
                #映画コーパスの各話者の発話データを読み込む
                for dat in jsonData:
                    #Turn1->Turn2
                    #userUttr=json.loads("".join(dat[u"turn_1"][:]))[u"actual_sentence"]
                    #sysUttr=json.loads("".join(dat[u"turn_2"][:]))[u"actual_sentence"]
                    #rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                    #Turn2->Turn3 リバースする
                    userUttr=json.loads("".join(dat[u"turn_2"][:]))[u"actual_sentence"]
                    userUttr=re.sub(u"\.+",u"",userUttr)
                    userUttr=re.sub(u" -+ ",u"",userUttr)
                    userUttr=re.sub(u"--",u"",userUttr)
                    userUttr=re.sub(u"-$",u"",userUttr)
                    userUttr=re.sub(u"- ",u"",userUttr)
                    userUttr=re.sub(u"-",u" ",userUttr)
                    userUttr=re.sub(u"\*",u"",userUttr)
                    userUttr=re.sub(u"\!+",u" !",userUttr)
                    userUttr=re.sub(u"\?+",u" ?",userUttr)
                    userUttr=re.sub(u"\"",u"",userUttr)
                    userUttr=re.sub(u"\[",u"",userUttr)
                    userUttr=re.sub(u"\]",u"",userUttr)
                    userUttr=re.sub(u",",u" ,",userUttr)
                    #number
                    userUttr=re.sub(u"([0-9])+",u"the value",userUttr)                    
                    userUttr=re.sub(u"([0-9])+:([0-9])+",u"the time",userUttr)
                    #shoryraku
                    userUttr=re.sub(u"'m",u" am",userUttr)
                    userUttr=re.sub(u"'re",u" are",userUttr)
                    userUttr=re.sub(u"'ll",u" will",userUttr)
                    #userUttr=re.sub(u"'s",u" is",userUttr)
                    userUttr=re.sub(u"'ve",u" have",userUttr)
                    #userUttr=re.sub(u"'d",u" would",userUttr)
                    userUttr=re.sub(u" '",u"",userUttr)
                    userUttr=re.sub(u"' ",u"",userUttr)
                    #space
                    userUttr=re.sub(u"^\s",u"",userUttr)
                    userUttr=re.sub(u"^ ",u"",userUttr)
                    userUttr=re.sub(u"\s+",u" ",userUttr)
                    userUttr=re.sub(u" +",u" ",userUttr)
                    userUttr=string.lower(userUttr)
                    
                    sysUttr=json.loads("".join(dat[u"turn_3"][:]))[u"actual_sentence"]
                    sysUttr=re.sub(u"\.+",u"",sysUttr)
                    sysUttr=re.sub(u" -+ ",u"",sysUttr)
                    sysUttr=re.sub(u"--",u"",sysUttr)
                    sysUttr=re.sub(u"-$",u"",sysUttr)
                    sysUttr=re.sub(u"- ",u"",sysUttr)
                    sysUttr=re.sub(u"-",u" ",sysUttr)
                    sysUttr=re.sub(u"\*",u"",sysUttr)
                    sysUttr=re.sub(u"\!+",u" !",sysUttr)
                    sysUttr=re.sub(u"\?+",u" ?",sysUttr)
                    sysUttr=re.sub(u"\"",u"",sysUttr)
                    sysUttr=re.sub(u"\[",u"",sysUttr)
                    sysUttr=re.sub(u"\]",u"",sysUttr)
                    sysUttr=re.sub(u",",u" ,",sysUttr)
                    #number
                    sysUttr=re.sub(u"([0-9])+",u"the value",sysUttr)                    
                    sysUttr=re.sub(u"([0-9])+:([0-9])+",u"the time",sysUttr)
                    #shoryaku
                    sysUttr=re.sub(u"'m",u" am",sysUttr)
                    sysUttr=re.sub(u"'re",u" are",sysUttr)
                    sysUttr=re.sub(u"'ll",u" will",sysUttr)
                    #sysUttr=re.sub(u"'s",u" is",sysUttr)
                    sysUttr=re.sub(u"'ve",u" have",sysUttr)
                    #sysUttr=re.sub(u"'d",u" would",sysUttr) 
                    sysUttr=re.sub(u" '",u"",sysUttr)
                    sysUttr=re.sub(u"' ",u"",sysUttr)
                    #space
                    sysUttr=re.sub(u"^\s",u"",sysUttr)
                    sysUttr=re.sub(u"^ ",u"",sysUttr)
                    sysUttr=re.sub(u"\s+",u" ",sysUttr)
                    sysUttr=re.sub(u" +",u" ",sysUttr)
                    sysUttr=string.lower(sysUttr)

                    rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                    #print rawIOLPairs
                #print "\n\n"
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()
            
            
            
            
            
            
    elif ExperimentalCondition.corpusType == "BusInfo":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(BusInfo)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*log\.json$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                call=json.load(f)
                prevSLURes=""
                prev2SLURes=""
                dictHist={}
                for turn in call["turns"]:
                    #print turn
                    sluRes=""
                    dictInput={}
                    for hyp in turn["input"]["batch"]["slu-hyps"]:
                        dictInput["DA_"+hyp["slu-hyp"][0]["act"]]=0
                        if len(turn["input"]["batch"]["slu-hyps"][0]["slu-hyp"][0]["slots"]) >0:
                            sVal="SV"
                            for val in turn["input"]["batch"]["slu-hyps"][0]["slu-hyp"][0]["slots"][0]:
                                sVal+="_"+str(val).replace(" ", "")
                            dictInput[sVal]=0
                    for key in dictInput.keys():
                        sluRes+=str(key)+" "
                    #Consider SLU results in most previous (i.e., user utterance)
                    #Determine input
                    userUttr=prevSLURes
                    for hist in dictHist.keys():
                        userUttr+= " H_"+hist
                    #Update History and Update input in most recenst
                    for pslu in prevSLURes.split(" "):
                        if (re.search("DA_.*",pslu) == None) and pslu != "":
                            dictHist[pslu]=0
                    prevSLURes=sluRes
                    sysUttr=""
                    if "transcript" in turn["output"]:
                        sysUttr=turn["output"]["transcript"]
                    else:
                        pass
                        #print turn["input"]["batch"]["slu-hyps"][0]
                        #print turn["output"]
                    userUttr=re.sub(u" +",u" ",userUttr)
                    
                    #print "User input: "+ userUttr
                    #print "Sys. output "+sysUttr
                    #print dictHist.keys()
                    #print ""
                    
                    rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                    #print rawIOLPairs
                    print "proc." + str(ind)
                    ind+=1
                #print "\n\n"
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()
    
    
    
    
    
    
    elif ExperimentalCondition.corpusType == "GuideDomain":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(GuideDomain)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*log\.json$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                call=json.load(f)
                userUttr=""
                sysUttr=""
                currentState="<S>"#it take value: "Guide" or "Tourist".
                for utterance in call["utterances"]:
                    #create user utterance and system response pair.
                    if currentState == "<S>":
                        if utterance["speaker"] == "Tourist":
                            userUttr+=" " + utterance["transcript"]
                            currentState="Tourist"
                        else:
                            continue
                    elif currentState == "Tourist":
                        if utterance["speaker"] == "Tourist":
                            userUttr+=" "+ utterance["transcript"]
                            currentState="Tourist"
                        elif utterance["speaker"] == "Guide":
                            sysUttr+=" " + utterance["transcript"]
                            currentState="Guide"
                    elif currentState == "Guide":
                        if utterance["speaker"] == "Tourist":
                            #pre-proc.
                            transt=re.sub("\,","",copy.copy(userUttr))
                            transt=re.sub("\?","",transt)
                            transt=re.sub("\.","",transt)
                            transt=re.sub("%","",transt)
                            transt=re.sub("(-|~)"," ",transt)
                            transt=re.sub("\!","",transt)
                            transt=re.sub("'"," ",transt)
                            transt=re.sub("\"","",transt)
                            transt=re.sub("/","",transt)
                            transt=re.sub("[1-9]+","replacedval",transt)
                            transt=re.sub(" +"," ",transt)
                            transt=re.sub(u"é","e",transt)
                            userUttr=transt.lower()
                            #--
                            transt=re.sub("\,","",copy.copy(sysUttr))
                            transt=re.sub("\?","",transt)
                            transt=re.sub("\.","",transt)
                            transt=re.sub("%","",transt)
                            transt=re.sub("(-|~)"," ",transt)
                            transt=re.sub("\!","",transt)
                            transt=re.sub("'"," ",transt)
                            transt=re.sub("\"","",transt)
                            transt=re.sub("/","",transt)
                            transt=re.sub("[1-9]+","replacedval",transt)
                            transt=re.sub(" +"," ",transt)
                            transt=re.sub(u"é","e",transt)
                            sysUttr=transt.lower()
                            #flush
                            print "proc." + str(ind)
                            print "User: "+userUttr
                            print "Sys: "+sysUttr
                            ind+=1
                            rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                            #
                            userUttr=utterance["transcript"]
                            sysUttr=""
                            currentState="Tourist"
                        elif utterance["speaker"] == "Guide":
                            sysUttr+=" " + utterance["transcript"]
                            currentState="Guide"                            
                #print "\n\n"
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("utf-8"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()

    



    elif ExperimentalCondition.corpusType == "Trains":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(Trains)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u"d(92|93).*", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                userUttr=""
                sysUttr=""
                currentState="<S>"#it take value: "User" or "System".
                prevSysUttr=""
                for utterance in f:
                    utterance=utterance.split(":")
                    print utterance
                    #create user utterance and system response pair.
                    if currentState == "<S>":
                        if len(utterance) >= 2:
                            if utterance[1] == " u":
                                userUttr+=" " + utterance[2]
                                currentState="User"
                            else:
                                continue
                    elif currentState == "User":
                        if (len(utterance)<3):
                            userUttr+=" "+ utterance[-1]
                            currentState="User"
                        elif (utterance[1] == " u"):
                            userUttr+=" "+ utterance[-1]
                            currentState="User"
                        elif utterance[1] == " s":
                            sysUttr+=" " + utterance[2]
                            currentState="System"
                        else:
                            assert False,"Unexpected pattern"
                    elif currentState == "System":
                        if (len(utterance)<3):
                            sysUttr+=" " + utterance[-1]
                            currentState="System"
                        elif utterance[1] == " u":
                            #pre-proc.
                            for word in prevSysUttr.split(" "):
                                if word != "":
                                    userUttr +=" sp_" + word
                            transt=re.sub("(<.*?>)","",copy.copy(userUttr))
                            transt=re.sub("\+","",transt)
                            transt=re.sub("\n","",transt)
                            transt=re.sub(" +"," ",transt)
                            userUttr=transt.lower()
                            #--
                            transt=re.sub("(<.*?>)?","",copy.copy(sysUttr))
                            transt=re.sub("\+","",transt)
                            transt=re.sub("\n","",transt)
                            transt=re.sub(" +"," ",transt)
                            sysUttr=transt.lower()
                            #flush
                            print "proc." + str(ind)
                            print "User: "+userUttr
                            print "Sys: "+sysUttr
                            ind+=1
                            rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                            #
                            userUttr=utterance[2]
                            prevSysUttr=sysUttr
                            sysUttr=""
                            currentState="User"
                        elif (utterance[1] == " u") or (len(utterance)<3):
                            sysUttr+=" " + utterance[-1]
                            currentState="System"
                        else:
                            assert False,"Unexpected pattern"
                #print "\n\n"
                rawDialog.append(rawIOLPairs)

            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("utf-8"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()


    elif ExperimentalCondition.corpusType == "Cleverbot":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(Cleverbot)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*\.log$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                userUttr=""
                sysUttr=""
                currentState="<S>"#it take value: "User" or "System".
                prevSysUttr=""
                for utterance in f:
                    utterance=utterance.split(":")
                    #print utterance
                    #create user utterance and system response pair.
                    if currentState == "<S>":
                        if len(utterance) != 1:
                            if utterance[0] == "User":
                                userUttr+=" " + utterance[1]
                                currentState="User"
                            else:
                                continue
                    elif currentState == "User":
                        if (len(utterance)==1):
                            userUttr=""
                            sysUttr=""
                            currentState="<S>"
                        elif utterance[0] == "User":
                            userUttr+=" "+ utterance[1]
                            currentState="User"
                        elif utterance[0] == "Cleverbot":
                            sysUttr+=" " + utterance[1]
                            currentState="Cleverbot"
                        else:
                            assert False,"Unexpected pattern"
                    elif currentState == "Cleverbot":
                        if (utterance[0] == "User") or (len(utterance)==1):
                            #pre-proc.
                            transt=re.sub("(<.*?>)","",copy.copy(userUttr))
                            transt=re.sub("\+"," + ",transt)
                            transt=re.sub("\*"," * ",transt)
                            transt=re.sub("\.+","",transt)
                            transt=re.sub("\?+"," ? ",transt)
                            transt=re.sub(",+","",transt)
                            transt=re.sub("!+"," ! ",transt)
                            transt=re.sub("\r","",transt)
                            transt=re.sub("\n","",transt)
                            transt=re.sub("\^","",transt)
                            transt=re.sub("\\\\","",transt)
                            transt=re.sub("_","",transt)
                            transt=re.sub("\(","",transt)
                            transt=re.sub("\)","",transt)
                            transt=re.sub("\"","",transt)
                            transt=re.sub("'","",transt)
                            transt=re.sub(" +"," ",transt)
                            #add
                            transt=transt.lower()
                            transt=re.sub("you're","you are",transt)
                            transt=re.sub("i'm","i am",transt)
                            transt=re.sub("he's","he is",transt)
                            transt=re.sub("she's","she is",transt)
                            transt=re.sub("they're","they are",transt)
                            #add2
                            transt=re.sub("you've","you have",transt)
                            transt=re.sub("i've","i have",transt)
                            transt=re.sub("they've","they have",transt)
                            #add3
                            transt=re.sub("can't","can not",transt)
                            transt=re.sub("couldn't","could not",transt)
                            transt=re.sub("'ll"," will",transt)
                            #Final
                            userUttr=transt
                            #--
                            transt=re.sub("(<.*?>)?","",copy.copy(sysUttr))
                            transt=re.sub("\+"," + ",transt)
                            transt=re.sub("\*"," * ",transt)
                            transt=re.sub("\.+","",transt)
                            transt=re.sub("\?+"," ? ",transt)
                            transt=re.sub(",+","",transt)
                            transt=re.sub("!+"," ! ",transt)
                            transt=re.sub("\r","",transt)
                            transt=re.sub("\n","",transt)
                            transt=re.sub("\^","",transt)
                            transt=re.sub("\\\\","",transt)
                            transt=re.sub("_","",transt)
                            transt=re.sub("\(","",transt)
                            transt=re.sub("\)","",transt)
                            transt=re.sub("\"","",transt)
                            transt=re.sub("'","",transt)
                            transt=re.sub(" +"," ",transt)
                            #add
                            transt=transt.lower()
                            transt=re.sub("you're","you are",transt)
                            transt=re.sub("i'm","i am",transt)
                            transt=re.sub("he's","he is",transt)
                            transt=re.sub("she's","she is",transt)
                            transt=re.sub("they're","they are",transt)
                            #add2
                            transt=re.sub("you've","you have",transt)
                            transt=re.sub("i've","i have",transt)
                            transt=re.sub("they've","they have",transt)
                            #add3
                            transt=re.sub("can't","can not",transt)
                            transt=re.sub("couldn't","could not",transt)
                            transt=re.sub("'ll"," will",transt)
                            #Final
                            sysUttr=transt
                            #flush
                            print "proc." + str(ind)
                            print "User: "+userUttr
                            print "Sys: "+sysUttr
                            ind+=1
                            rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                            #
                            if (utterance[0] == "User"):
                                userUttr=utterance[1]
                                currentState="User"
                            else:
                                userUttr=""
                                currentState="<S>"
                            sysUttr=""
                        elif (utterance[0] == "Cleverbot"):
                            sysUttr+=" " + utterance[1]
                            currentState="Cleverbot"
                        else:
                            assert False,"Unexpected pattern"
                    else:
                        assert False,"Unexpected pattern"
                #print "\n\n"
                rawDialog.append(rawIOLPairs)

            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("utf-8"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()
    
    
    
    
    
    
    
    
    elif ExperimentalCondition.corpusType == "RestrantInfo":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(RestrantInfo)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*log\.json$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                call=json.load(f)
                prevSLURes=""
                prev2SLURes=""
                dictHist={}
                for turn in call["turns"]:
                    #print turn
                    sluRes=""
                    dictInput={}
                    for hyp in turn["input"]["live"]["slu-hyps"]:
                        if len(hyp["slu-hyp"])>0:
                            dictInput["DA_"+hyp["slu-hyp"][0]["act"]]=0
                            if len(hyp["slu-hyp"][0]["slots"]) >0:
                                sVal="SV"
                                for val in hyp["slu-hyp"][0]["slots"][0]:
                                    sVal+="_"+str(val).replace(" ", "")
                                dictInput[sVal]=0
                    for key in dictInput.keys():
                        sluRes+=str(key)+" "
                    #Consider SLU results in most previous (i.e., user utterance)
                    #Determine input
                    userUttr=prevSLURes
                    for hist in dictHist.keys():
                        userUttr+= " H_"+hist
                    #Update History and Update input in most recenst
                    for pslu in prevSLURes.split(" "):
                        if (re.search("DA_.*",pslu) == None) and pslu != "":
                            dictHist[pslu]=0
                    prevSLURes=sluRes
                    sysUttr=""
                    if "transcript" in turn["output"]:
                        sysUttr=turn["output"]["transcript"]
                    else:
                        pass
                        #print turn["input"]["batch"]["slu-hyps"][0]
                        #print turn["output"]
                    userUttr=re.sub(u" +",u" ",userUttr)
                    
                    #print "User input: "+ userUttr
                    #print "Sys. output "+sysUttr
                    #print dictHist.keys()
                    #print ""
                    
                    rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                    #print rawIOLPairs
                    print "proc." + str(ind)
                    ind+=1
                #print "\n\n"
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()

    
    
    
    


    elif ExperimentalCondition.corpusType == "TouristInfo":#************************************
        if ExperimentalCondition.isInitialyCreateCompiledUserSystemUteranceAndVecorizor:
            print u"オラクルの対話ファイルの読み込み(TouristInfo)"
            allDialogFiles=[]#発話を集めたもの
            directories=[u".\OracleCorpus"]
            while len(directories) > 0:
                currentDir=directories.pop()
                expandedFiles=glob.glob(currentDir+u"/*")
                for file in expandedFiles:
                    if os.path.isdir(file):
                        directories.append(file)
                    elif re.search(u".*log\.json$", file) != None:
                        allDialogFiles.append(file)
            #print allDialogFiles
            #print len(allDialogFiles)
            
            rawDialog=[]#全対話->　対話=[<入力-応答,ラベル>...]
            ind=0
            for dialogFile in allDialogFiles:
                f=codecs.open(dialogFile)#,u"r",u"utf-8")
                rawIOLPairs=[]
                #バスコーパスの各話者の発話データを読み込む
                call=json.load(f)
                prevSLURes=""
                prev2SLURes=""
                dictHist={}
                for turn in call["turns"]:
                    #print turn
                    sluRes=""
                    dictInput={}
                    for hyp in turn["input"]["live"]["slu-hyps"]:
                        if len(hyp["slu-hyp"])>0:
                            dictInput["DA_"+hyp["slu-hyp"][0]["act"]]=0
                            if len(hyp["slu-hyp"][0]["slots"]) >0:
                                sVal="SV"
                                for val in hyp["slu-hyp"][0]["slots"][0]:
                                    sVal+="_"+str(val).replace(" ", "")
                                dictInput[sVal]=0
                    for key in dictInput.keys():
                        sluRes+=str(key)+" "
                    #Consider SLU results in most previous (i.e., user utterance)
                    #Determine input
                    userUttr=prevSLURes
                    for hist in dictHist.keys():
                        userUttr+= " H_"+hist
                    #Update History and Update input in most recenst
                    for pslu in prevSLURes.split(" "):
                        if (re.search("DA_.*",pslu) == None) and pslu != "":
                            dictHist[pslu]=0
                    prevSLURes=sluRes
                    sysUttr=""
                    if "transcript" in turn["output"]:
                        sysUttr=turn["output"]["transcript"]
                    else:
                        pass
                        #print turn["input"]["batch"]["slu-hyps"][0]
                        #print turn["output"]
                    userUttr=re.sub(u" +",u" ",userUttr)
                    
                    #print "User input: "+ userUttr
                    #print "Sys. output "+sysUttr
                    #print dictHist.keys()
                    #print ""
                    
                    rawIOLPairs.append(["".join(userUttr), "".join(sysUttr)])
                    #print rawIOLPairs
                    print "proc." + str(ind)
                    ind+=1
                #print "\n\n"
                rawDialog.append(rawIOLPairs)
            #--誤りが含まれるシステムの発話は除外
            print u"システムが不適切な応答をしている部分の削除"
            rawDialogWithoutIncorrectResponse=[]
            for i in range(len(rawDialog)):
                tempPair=[]
                for j in range(len(rawDialog[i])):
                    if rawDialog[i][j][1] !="" and (rawDialog[i][j][0] != "") and (rawDialog[i][j][1] !=" ") and (rawDialog[i][j][0] != " "):
                        tempPair.append(rawDialog[i][j])
                rawDialogWithoutIncorrectResponse.append(tempPair)
            #用例のベクトル表現のコンパイル
            print u"読み込んだ対話データのコンパイル"
            w2vComp=Word2VecCompiler()
            print u"構築"
            print u"ユーザ発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][0].encode("shift-jis"))
            vectorizorForUserUtterane, compiledUserInput=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたユーザの発話
            fPkl=gzip.open("CompiledUserUtterance.pkl","wb")
            pickle.dump(compiledUserInput,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4User.pkl","wb")
            vectorizorForUserUtterane.analyzer=None
            pickle.dump(vectorizorForUserUtterane, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #####
            print u"システム発話のコンパイル"
            dialogs=[]
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    dialogs.append(rawDialogWithoutIncorrectResponse[i][j][1].encode("shift-jis"))
            vectorizorForSystemUtterance, compiledSystemresponce=w2vComp.ConstructCimilarityCalculatorAndTfIDFVectors(dialogs)
            #コンパイルされたシステムの発話
            fPkl=gzip.open("CompiledSystemUtterance.pkl","wb")
            pickle.dump(compiledSystemresponce,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            #Vectorizor
            fPkl=gzip.open("Vectorizor4System.pkl","wb")
            vectorizorForSystemUtterance.analyzer=None
            pickle.dump(vectorizorForSystemUtterance, fPkl,pickle.HIGHEST_PROTOCOL)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            
            #コンパイルつきのデータを作成
            indUser=0
            indSys=0
            for i in range(len(rawDialogWithoutIncorrectResponse)):
                for j in range(len(rawDialogWithoutIncorrectResponse[i])):
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledUserInput[indUser])
                    rawDialogWithoutIncorrectResponse[i][j].append(compiledSystemresponce[indSys])
                    indUser+=1
                    indSys+=1
                    #print rawDialogWithoutIncorrectResponse[i][j][0] + u"->"+ rawDialogWithoutIncorrectResponse[i][j][1]
            #コンパイルつきのデータ
            dialogWithCompileInfo=copy.deepcopy(rawDialogWithoutIncorrectResponse)
            fPkl=gzip.open("DialogWithCompileInfo.pkl","wb")
            pickle.dump(dialogWithCompileInfo,fPkl,pickle.HIGHEST_PROTOCOL)
            fPkl.close()
            
        else:
            print u"Pickleファイルから読み込み"
            #
            fPkl=gzip.open("CompiledUserUtterance.pkl","rb")
            compiledUserInput=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("CompiledSystemUtterance.pkl","rb")
            compiledSystemresponce=pickle.load(fPkl)
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4User.pkl","rb")
            vectorizorForUserUtterane=pickle.load(fPkl)
            vectorizorForUserUtterane.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("Vectorizor4System.pkl","rb")
            vectorizorForSystemUtterance=pickle.load(fPkl)
            vectorizorForSystemUtterance.analyzer=w2vComp.stems
            fPkl.close()
            #
            fPkl=gzip.open("DialogWithCompileInfo.pkl","rb")
            dialogWithCompileInfo=pickle.load(fPkl)
            fPkl.close()



    
    
    
    
    
    
    
    
    else:#***************************************************************************************
        assert False, "Illeagal Corpus type"        
        
        
        
    print "TotalLen="+str(len(dialogWithCompileInfo))
    #テストセットをあらかじめOracleから間引く
    testSet=[]#
    sizeOfTestSet=0
    rand4TestSet=random.RandomState(1192)#一様にテストデータを選択するように設定
    while sizeOfTestSet < ExperimentalCondition.num4HoldoutTestSet:
        #oracleDial=dialogWithCompileInfo.pop()
        #testSet.append(oracleDial)
        #sizeOfTestSet+=len(oracleDial)
        iDial=rand4TestSet.randint(low=0,high=99999)%len(dialogWithCompileInfo)
        if len(dialogWithCompileInfo[iDial]) <= 0:
            dialogWithCompileInfo.pop(iDial)
        else:
            iPair=rand4TestSet.randint(low=0,high=99999)%len(dialogWithCompileInfo[iDial])
            oracleDial=dialogWithCompileInfo[iDial].pop(iPair)
            testSet.append([oracleDial])
            sizeOfTestSet+=1
            if len(dialogWithCompileInfo[iDial]) <= 0:
                dialogWithCompileInfo.pop(iDial)
    #print len(testSet)
    #for i in range(len(testSet)): 
    #    print testSet[i][0][0]
    random.seed()#時間に依存させる
    
    #実験設定表示
    if ExperimentalCondition.creationMethod == "Random":
        print "Method:Random"
    elif ExperimentalCondition.creationMethod == "MinInExample":
        print "Method:MinInExample"
    elif ExperimentalCondition.creationMethod == "AvrMaxInPool":
        print "Method:AvrMaxInPool"    
    elif ExperimentalCondition.creationMethod == "ArithMinAndAvrMax":
        print "Method:ArithMinAndAvrMax"
    elif ExperimentalCondition.creationMethod == "GeoMinAndAvrMax":
        print "Method:GeoMinAndAvrMax"
    elif ExperimentalCondition.creationMethod == "DWArithMinAndAvrMax":
        print "Method:DWArithMinAndAvrMax"
        print "WeightDecay 1/" + str(ExperimentalCondition.weightDecayOfAvrMaxInPoolAtEachStep)
    elif ExperimentalCondition.creationMethod == "CWArithMinAndAvrMax":
        print "Method:CWArithMinAndAvrMax"
    elif ExperimentalCondition.creationMethod == "ESysMinSimInExample":
        print "Method: ESysMinSimInExample"
    elif ExperimentalCondition.creationMethod == "ESysAvrMaxInPool":
        print "Method:ESysAvrMaxInPool"
    else:
        assert False, "Illegeal experimentalCondition creationMethod"
         
    if ExperimentalCondition.isInverseScore:
        print "Use Inverse Criterion"
    else:
        print "Normal Order"
    if ExperimentalCondition.isIgnoreOverlappedQuery:
        print "Ignore query which already queried in previous"
    else:
        print "Append query even if the query is already queried in previous. "
    if ExperimentalCondition.isInverseWeight:
        print "The weight of CWArith or CWGeo is inverted"
    if ExperimentalCondition.isSamplingQueriesBasedOnScore:
        print "If MinInExample and AvrMaxInPool are used. Queries are sampled baswd on the score. "
    
    numSystem=0
    numStep=0
    averageOverSystems=[]
    #高速化用キャッシュ
    similarityUserUtteranceInExampleBaseAndInOracle={}#[オラクルのユーザ発話][システムの用例中のユーザ発話]＝類似度
    similaritySystemUtteranceFromSystemBaseAndFromOracle={}#[システムが出力した発話][オラクル中のシステム発話]＝類似度
    while (numSystem < ExperimentalCondition.numberOfMaxSystem):
        #高速化用キャッシュ
        bestSimAndExamplePairTowardOracle={}#[オラクルのユーザ発話]=[もっとも類似しているシステムの発話のスコア,もっとも類似しているシステム発話]

        #初期化
        d=datetime.datetime.today()
        f=open(str(numSystem)+"Experiment"+str(d.year)+str(d.month+d.day)+"_"+str(d.hour)+str(d.minute)+".txt","w")
        print "System:"+str(numSystem)
        averageOverSystems.append([])
        #-読み込んだオラクルの対話コーパスを基にして対話システムの初期用例の構築。初期用例数に応じて、オラクルのコーパスの一部を用例として割り当てる
        print u"システムの初期用例の構築"
        exampleDataBase=[]#システムの用例：D=[P..],P=<user utterance,system utterance, Compiled user utter,Compiled sys utterance>
        remainingOracle=[]#システムの用例に含まれないオラクルのデータ：D=[P..],P=<user utterance,system utterance, Compiled user utter,Compiled sys utterance>
        sizeOfDB=0.0
        remainingOracle=[]
        for dialog in dialogWithCompileInfo:
            for pair in dialog:
                remainingOracle.append(pair)
        #昔の実験ファイルと条件を揃えるためのLegacy code
        print compiledUserInput.get_shape()
        print "LENGTH:" + str(float(compiledUserInput.getnnz())*ExperimentalCondition.percentile4InitialExampleDatabaseSize)
        while float(sizeOfDB) < (float(compiledUserInput.getnnz())*ExperimentalCondition.percentile4InitialExampleDatabaseSize):
        #New code after V20
        #tSize=float(len(remainingOracle))
        #while float(sizeOfDB) < (tSize*ExperimentalCondition.percentile4InitialExampleDatabaseSize):
            exampleDataBase.append(remainingOracle.pop(random.randint(low=0,high=99999)%len(remainingOracle)))
            sizeOfDB+=1
        #初期のテストセット評価：testSetに対して生成されたシステムの返答とオラクルのシステムの返答の平均コサイン類似度を評価
        avrSimSystemRespVSOracle=0.0
        i=0
        #Worstスコアの計算
        worstScore=0.0        
        for dial in testSet:
            maxSim4Test=-1.0
            maxResp4Test=None
            for oracle in dial:
                for examplePair in exampleDataBase:
                    if oracle[0] not in similarityUserUtteranceInExampleBaseAndInOracle:
                        similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]]={}
                        similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    else:
                        if examplePair[0] not in similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]]:
                            similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                    sim=similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]
                    
                    if sim > maxSim4Test:
                        maxSim4Test=sim
                        maxResp4Test=examplePair
                bestSimAndExamplePairTowardOracle[oracle[0]]=[maxSim4Test,maxResp4Test]
                #
                if maxResp4Test[1] not in similaritySystemUtteranceFromSystemBaseAndFromOracle:
                    similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]]={}
                    similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]=cosine_similarity(bestSimAndExamplePairTowardOracle[oracle[0]][1][3], oracle[3])[0][0]*10000.0
                else:
                    if oracle[1] not in similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]]:
                        similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]=cosine_similarity(bestSimAndExamplePairTowardOracle[oracle[0]][1][3], oracle[3])[0][0]*10000.0
                avrSimSystemRespVSOracle+=similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]
                if worstScore > similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]:
                    worstScore=similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]
                i+=1
        avrSimSystemRespVSOracle/=float(i)
        print str(numStep)+"Average cosine sim.(*10000)=" + str(avrSimSystemRespVSOracle)
        f.write(str(numStep)+"Average cosine sim.(*10000)=" + str(avrSimSystemRespVSOracle)+"\n")
        averageOverSystems[numSystem].append(avrSimSystemRespVSOracle)
        print str(numStep)+"worstScore=" + str(worstScore)

        
        #既定のステップが経過するまでアクティブラーニングを繰り返す    
        numStep=0
        dicNumEqualSelection={}#テストデータに対する同一評価値を持つシステム応答文を選択した回数
        #手法作成
        examplePairCreator=ExamplePairsCreator()
        examplePairCrationMethod=None
        if ExperimentalCondition.creationMethod == "Random":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_random
        elif ExperimentalCondition.creationMethod == "MinInExample":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_MinSimInExample
        elif ExperimentalCondition.creationMethod == "AvrMaxInPool":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_AvrMaxInPool
        elif ExperimentalCondition.creationMethod == "ArithMinAndAvrMax":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_ArithmeticMinSimInExampleAndAvrMaxSimInPool
        elif ExperimentalCondition.creationMethod == "GeoMinAndAvrMax":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_GeometrixMinSimInExampleAndAvrMaxSimInPool
        elif ExperimentalCondition.creationMethod == "DWArithMinAndAvrMax":
            examplePairCrationMethod=examplePairCreator.creatExamplePairs_DWArithmeticMinSimInExampleAndAvrMaxSimInPool
        elif ExperimentalCondition.creationMethod == "CWArithMinAndAvrMax":
            examplePairCrationMethod=examplePairCrationMethod=examplePairCreator.creatExamplePairs_CWArithMinAndAvrMax
        elif ExperimentalCondition.creationMethod == "ESysMinSimInExample":
            examplePairCrationMethod=examplePairCrationMethod=examplePairCreator.creatExamplePairs_ESysMinSimInExample
        elif ExperimentalCondition.creationMethod == "ESysAvrMaxInPool":
            examplePairCrationMethod=examplePairCrationMethod=examplePairCreator.creatExamplePairs_ESysAvrMaxInPool
        else: 
            assert False, "Illegeal experimentalCondition creationMethod"
        while (numStep < ExperimentalCondition.numberOfMaxStep) and (len(remainingOracle)>0):
            #-ラベリング・学習フェーズ
            createdExample, remainingOracle=examplePairCrationMethod(exampleDataBase, remainingOracle)
            for elem in createdExample:
                exampleDataBase.append(elem)
            if ExperimentalCondition.isTraceQuery:#V13で追加
                print "Query at " + str(numStep) + ":"
                for pair in createdExample:
                    print ","+pair[0],
                print ""
                
            #-評価フェーズ：testSetに対して生成されたシステムの返答とオラクルのシステムの返答の平均コサイン類似度を評価
            avrSimSystemRespVSOracle=0.0
            i=0
            #Worstスコアの計算
            worstScore=0.0
            for dial in testSet:
                maxSim4Test=-1.0
                maxResp4Test=None
                for oracle in dial:
                    for examplePair in createdExample:
                        if oracle[0] not in similarityUserUtteranceInExampleBaseAndInOracle:
                            similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]]={}
                            similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                        else:
                            if examplePair[0] not in similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]]:
                                similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]=cosine_similarity(oracle[2], examplePair[2])[0][0]
                        sim=similarityUserUtteranceInExampleBaseAndInOracle[oracle[0]][examplePair[0]]
                        
                        if sim > maxSim4Test:
                            maxSim4Test=sim
                            maxResp4Test=examplePair
                        elif sim == maxSim4Test:#同じスコア持つ候補が存在した場合、ランダムに更新
                            probUpdate=None
                            if oracle[0] not in dicNumEqualSelection:
                                dicNumEqualSelection[oracle[0]]=2.0
                            else:
                                dicNumEqualSelection[oracle[0]]+=1.0
                            probUpdate=1.0/dicNumEqualSelection[oracle[0]]
                            #print str(probUpdate)
                            if random.rand() < probUpdate:
                                #print "Swapped"
                                maxSim4Test=sim
                                maxResp4Test=examplePair
                    if maxSim4Test > bestSimAndExamplePairTowardOracle[oracle[0]][0]:
                        bestSimAndExamplePairTowardOracle[oracle[0]]=[maxSim4Test,maxResp4Test]
                    else:
                        maxSim4Test=bestSimAndExamplePairTowardOracle[oracle[0]][0]
                        maxResp4Test=bestSimAndExamplePairTowardOracle[oracle[0]][1]
                    #
                    if maxResp4Test[1] not in similaritySystemUtteranceFromSystemBaseAndFromOracle:
                        similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]]={}
                        similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]=cosine_similarity(bestSimAndExamplePairTowardOracle[oracle[0]][1][3], oracle[3])[0][0]*10000.0
                    else:
                        if oracle[1] not in similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]]:
                            similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]=cosine_similarity(bestSimAndExamplePairTowardOracle[oracle[0]][1][3], oracle[3])[0][0]*10000.0
                    avrSimSystemRespVSOracle+=similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]
                    if worstScore > similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]:
                        worstScore=similaritySystemUtteranceFromSystemBaseAndFromOracle[maxResp4Test[1]][oracle[1]]
                    i+=1
            avrSimSystemRespVSOracle/=float(i)
            print str(numStep)+"Average cosine sim.(*10000)=" + str(avrSimSystemRespVSOracle)
            f.write(str(numStep)+"Average cosine sim.(*10000)=" + str(avrSimSystemRespVSOracle)+"\n")
            averageOverSystems[numSystem].append(avrSimSystemRespVSOracle)
            print str(numStep)+"worstScore=" + str(worstScore)
            
            #End of each step
            numStep+=1
        #End of each system 
        f.close()
        numSystem+=1
    #End of method
    #これまでの手法の計算で各ステップごとの平均を計算
    step=[]
    for j in range(numSystem):
            step.append(len(averageOverSystems[j]))
    numStep=min(step)

    averagePerStep=[]
    for i in range(numStep):
        ave=0.0
        for j in range(numSystem):
            ave+=averageOverSystems[j][i]
        ave/=float(numSystem)
        averagePerStep.append(ave)
        print str(i)+"Average over system="+ str(ave)
    #分散を計算
    for i in range(numStep):
        var=0.0
        for j in range(numSystem):
            var+=(averageOverSystems[j][i]-averagePerStep[i])*(averageOverSystems[j][i]-averagePerStep[i])
        var/=float(numSystem)
        print str(i)+"Variance over system="+ str(var)
        