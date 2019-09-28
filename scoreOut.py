# -*- coding: utf-8 -*-
import urllib,urllib2,datetime,decimal,traceback,time,os,glob,gzip,re
import jieba
# import jieba.posseg as pseg
import pickle

#导入各个情感词典库
posdict = pickle.load(open(r'dic\positive.pkl', 'r'))
overdict = pickle.load(open(r'dic\over.pkl', 'r'))
negdict = pickle.load(open(r'dic\negative.pkl', 'r'))
moredict = pickle.load(open(r'dic\more.pkl', 'r'))
mostdict = pickle.load(open(r'dic\most.pkl', 'r'))
overdict = pickle.load(open(r'dic\over.pkl', 'r'))
verydict = pickle.load(open(r'dic\very.pkl', 'r'))
insufficientdict = pickle.load(open(r'dic\ins.pkl', 'r'))
ishdict = pickle.load(open(r'dic\ish.pkl', 'r'))
bandict = pickle.load(open(r'dic\ban.pkl', 'r'))
inversedict = pickle.load(open(r'dic\inverse.pkl', 'r'))
#判断基偶数（否定时用）
def judgeodd(num):
    if (num/2)*2 == num:
        return 'even'
    else:
        return 'odd'

#文本切割并且去禁用词
def sent2word(sentence):
    segList = jieba.cut(sentence)
    segResult = []
    for w in segList:
        segResult.append(w.encode('utf-8'))
    newSent = []
    for word in segResult:
        if word in bandict:
            # print "stopword: %s" % word
            continue
        else:
            newSent.append(word)
    return newSent


#对分词进行情感算分
def sentiment_score_list(segtmp):

    count1 = []
    count2 = []
    i = 0 #记录扫描到的词的位置
    a = 0 #记录情感词的位置
    poscount = 0 #积极词的第一次分值
    poscount2 = 0 #积极词反转后的分值
    poscount3 = 0 #积极词的最后分值（包括叹号的分值）
    negcount = 0
    negcount2 = 0
    negcount3 = 0
    for word in segtmp:
        if word in posdict:
            poscount += 1
            c = 0  #反转词
            for w in segtmp[a:i]:  #扫描情感词前的程度词
                if w in mostdict:
                    poscount *= 4.0
                elif w in verydict:
                    poscount *= 3.0
                elif w in moredict:
                    poscount *= 2.0
                elif w in ishdict:
                    poscount /= 2.0
                elif w in insufficientdict:
                    poscount /= 4.0
                elif w in inversedict:
                    c += 1
            if judgeodd(c) == 'odd': #扫描情感词前的否定词数
                poscount *= -1.0
                poscount2 += poscount
                poscount = 0
                poscount3 = poscount + poscount2 + poscount3
                poscount2 = 0
            else:
                poscount3 = poscount + poscount2 + poscount3
                poscount = 0
            a = i + 1 #情感词的位置变化
        elif word in negdict: #消极情感的分析，与上面一致
            negcount += 1
            d = 0
            for w in segtmp[a:i]:
                if w in mostdict:
                    negcount *= 4.0
                elif w in verydict:
                    negcount *= 3.0
                elif w in moredict:
                    negcount *= 2.0
                elif w in ishdict:
                    negcount /= 2.0
                elif w in insufficientdict:
                    negcount /= 4.0
                elif w in inversedict:
                    d += 1
            if judgeodd(d) == 'odd':
                negcount *= -1.0
                negcount2 += negcount
                negcount = 0
                negcount3 = negcount + negcount2 + negcount3
                negcount2 = 0
            else:
                negcount3 = negcount + negcount2 + negcount3
                negcount = 0
            a = i + 1
        elif word == '！' or word == '!': ##判断句子是否有感叹号
            for w2 in segtmp[::-1]: #扫描感叹号前的情感词，发现后权值+2，然后退出循环
                if w2 in posdict or negdict:
                    if poscount3 != 0:poscount3 += 2
                    poscount3 += 2
                    if negcount3 != 0:negcount3 += 2
                    break
        i += 1 #扫描词位置前移
    #对负数进行处理
    pos_count = 0
    neg_count = 0
    if poscount3 < 0 and negcount3 > 0:
        neg_count += negcount3 - poscount3
        pos_count = 0
    elif negcount3 < 0 and poscount3 > 0:
        pos_count = poscount3 - negcount3
        neg_count = 0
    elif poscount3 < 0 and negcount3 < 0:
        neg_count = -poscount3
        pos_count = -negcount3
    else:
        pos_count = poscount3
        neg_count = negcount3
    count1.append([pos_count, neg_count])
    return count1






