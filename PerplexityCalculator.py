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
import ngrams
#20150814追加

#実験条件設定用変数
class ExperimentalCondition:
    isInitialyCreateCompiledUserSystemUteranceAndVecorizor=True#実験用データをファイルから構築しなおすか
    isUseTfIDFweight=True#実験用データをファイルから構築しなおす際に、tf-idf重み付けを利用するか
    
    #対象データに対する実験変数
    corpusType="ProjectNextNLP"
    #ProjectNextNLP
    #Switchboard
    #IdosMovie
    #BusInfo
    #GuideDomain
    #Trains
    #Cleverbot
    #RestrantInfo
    #TouristInfo
    
    percentileOfFoldout=0.6#How many data are used for test
    
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
        #print u"Num"+str(self.numIteration)+":"+text
        print u""+text
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
            #print allDialogFiles
            
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
        
    comp=Word2VecCompiler()
    testUser=[]
    trainUser=[]
    testSys=[]
    trainSys=[]
    totallen=0.0
    for dial in dialogWithCompileInfo:
        totallen+=len(dial)
    
    ind=0
    for dial in dialogWithCompileInfo:
        for pair in dial:
            if ((float)(totallen)*ExperimentalCondition.percentileOfFoldout) > ind:
                if ExperimentalCondition.corpusType=="ProjectNextNLP":#V12で修正
                    trainUser.append(comp.stems(pair[0].encode("shift-jis")))
                    trainSys.append(comp.stems(pair[1].encode("shift-jis")))
                else:
                    trainUser.append(comp.split_EnglishSentence(pair[0]))
                    trainSys.append(comp.split_EnglishSentence(pair[1]))
                    print comp.split_EnglishSentence(pair[0])
                    print comp.split_EnglishSentence(pair[1])

            else:
                if ExperimentalCondition.corpusType=="ProjectNextNLP":#V12で修正
                    testUser.append(comp.stems(pair[0].encode("shift-jis")))
                    testSys.append(comp.stems(pair[1].encode("shift-jis")))
                else:
                    testUser.append(comp.split_EnglishSentence(pair[0]))
                    testSys.append(comp.split_EnglishSentence(pair[1]))
                    print comp.split_EnglishSentence(pair[0])
                    print comp.split_EnglishSentence(pair[1])

            ind+=1
    
    #Separately calculate
    f=open("trainUser.txt","wb")
    for tex in trainUser:
        buf=""
        for char in tex:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
 
    f=open("testUser.txt","wb")
    for tex in testUser:
        buf=""
        for char in tex:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
     
    f=open("trainSys.txt","wb")
    for tex in trainSys:
        buf=""
        for char in tex:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
     
    f=open("testSys.txt","wb")
    for tex in testSys:
        buf=""
        for char in tex:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
 
    print "Perplexity User:"
    #print os.system("python ngrams.py -p trainUser.txt testUser.txt")
    print os.system("java -cp kylm.jar kylm.main.CountNgrams -smoothuni -n 1 trainUser.txt model.arpa")
    print os.system("java -cp kylm.jar kylm.main.CrossEntropy  -arpa model.arpa testUser.txt")
    
    
    print "\n\n Perplexity Sys:"
    #print os.system("python ngrams.py -p trainSys.txt testSys.txt")
    print os.system("java -cp kylm.jar kylm.main.CountNgrams -smoothuni -n 1  trainSys.txt model.arpa")
    print os.system("java -cp kylm.jar kylm.main.CrossEntropy  -arpa model.arpa testSys.txt")
    
    
    
    







    #Jontly calculate
    f=open("train.txt","wb")
    for ind in range(len(trainUser)):
        buf=""
        for char in trainUser[ind]:
            buf+=" " + char
        for char in trainSys[ind]:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
         
    f=open("test.txt","wb")
    for ind in range(len(testUser)):
        buf=""
        for char in testUser[ind]:
            buf+=" " + char
        for char in testSys[ind]:
            buf+=" " + char
        buf+="\n"
        f.write(buf)
    f.close()
    
    print "\n\nPerplexity Joint"
    #print os.system("python ngrams.py -p train.txt test.txt")
    print os.system("java -cp kylm.jar kylm.main.CountNgrams -smoothuni -n 1 train.txt model.arpa")
    print os.system("java -cp kylm.jar kylm.main.CrossEntropy  -arpa model.arpa test.txt")
