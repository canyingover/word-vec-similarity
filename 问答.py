#!/usr/bin/env python
# coding: utf-8

# ### TF-IDF版本

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import pandas as pd
jieba.load_userdict(u"C:/Users/n4663/Desktop/精灵相似匹配项目/yysword")
data = pd.read_csv(u"C:/Users/n4663/Desktop/精灵相似匹配项目/test.txt",sep="\t",encoding="utf-8")


# In[2]:


data.columns


# In[3]:


# seg_list = jieba.cut("我来了,北京里面的清华大学", cut_all=False)


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:



#groupbydata2 = pd.DataFrame({'count' : data2.groupby(['role_id']).size()}).reset_index()
groupbydata = pd.DataFrame({"count":data.groupby([u"精灵答案"]).size()}).reset_index()


# In[7]:


data = pd.merge(data,groupbydata,on=u"精灵答案").loc[:40000,:]


# In[8]:


data.shape


# In[9]:


segment_jieba = lambda text: " ".join(jieba.cut(text))
corpus = []
for i in data.loc[:,u"用户请求"]:
    corpus.append(segment_jieba(i))


# In[10]:


#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
# X = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))


# In[11]:


# word = vectorizer.get_feature_names()
tfidf_weight = tfidf.toarray()


# In[12]:


len(tfidf_weight[0])


# In[13]:


import sys
print sys.getsizeof(tfidf_weight)


# In[14]:


# def cosine_similarity2(vector1, vector2):
#     dot_product = 0.0
#     normA = 0.0
#     normB = 0.0
#     for a, b in zip(vector1, vector2):
#         dot_product += a * b
#         normA += a ** 2
#         normB += b ** 2
#     if normA == 0.0 or normB == 0.0:
#         return 0
#     else:
#         return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


# In[15]:


# from sklearn.metrics.pairwise import cosine_similarity

# word_test1 = u"满级晴明属性是多少"
# # word_test2 = u"晴明满级属性"
# corpus_test1 = []
# corpus_test1.append(segment_jieba(word_test1))

# vec_test1 = vectorizer.transform(corpus_test1)
# TF_test1 = transformer.transform(vec_test1)

# # vec_test2 = vectorizer.transform([word_test2])
# # TF_test2 = transformer.transform(vec_test2)
# #TF_test.toarray()[0][0] == tfidf_weight[0][0]
# #list(TF_test.toarray()[0])
# #tfidf_weight[0]
# print word_test1
# print list(data.loc[:,u"用户请求"])[0]


# cosine_similarity([list(TF_test1.toarray()[0]),list(tfidf_weight[0])])[0][1]

# #cosine_similarity2(list(TF_test1.toarray()[0]),list(tfidf_weight[0]))


# In[16]:



# {"精灵答案":[[用户请求1(向量)],[用户请求2],[用户请求3]]}
import numpy as np
label_dict = {}
for index,each in enumerate(data.itertuples()):
    key = each[5]
    label_dict.setdefault(key,{"vector":[],"ques":[]})

#     kk = ','.join(map(lambda x:str(x),list(tfidf_weight[index])))

#     label_dict[key].append(kk)
    label_dict[key]["vector"].append(list(tfidf_weight[index]))
    label_dict[key]["ques"].append(corpus[index])
    
    


# In[17]:


# for each in label_dict:
#     print each
#     print label_dict[each]
#     break


# In[18]:


# {"精灵答案":"中心点向量"}
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
label_center={}
for i in label_dict:
    X = label_dict[i]["vector"]
    label_center.setdefault(i,{"all_ques":label_dict[i]["ques"],"center":[],"nearest":[],"max_score":0})    
    y_pred = KMeans(n_clusters=1, random_state=9).fit(X)
    label_center[i]["center"] = y_pred.cluster_centers_[0]
    max_ = 0
    for index,j in enumerate(label_dict[i]["vector"]):
        degree = cosine_similarity([label_center[i]["center"],j])[0][1]
        if degree > max_:
            max_ = degree
            label_center[i]["nearest"] = label_dict[i]["ques"][index]
            label_center[i]["max_score"] = max_

            
        #     print y_pred.cluster_centers_[0]



# In[19]:


import codecs
import json
result =pd.DataFrame(columns=('ans','ques','nearest','max_score'))
with codecs.open(u"C:/Users/n4663/Desktop/精灵相似匹配项目/result.txt",'w',encoding='utf-8') as w:
    for i in label_center:
        ans = unicode(i)
        ques = unicode(json.dumps(label_center[i]["all_ques"],ensure_ascii=False))
        nearest = unicode(label_center[i]["nearest"])
        max_score = str(label_center[i]["max_score"])
        result=result.append(pd.DataFrame({'ans':[ans],'ques':[ques],'nearest':[nearest],'max_score':[max_score]}),ignore_index=True)
        
        
        record = '\t'.join([unicode(i),unicode(json.dumps(label_center[i]["all_ques"],ensure_ascii=False)),unicode(label_center[i]["nearest"]),str(label_center[i]["max_score"])]) + '\n'
        w.write(record)
#         print label_center[i]["center"]
#         print label_center[i]["nearest"]
#         print label_center[i]["all_ques"]
#         print label_center[i]["max_score"]
#         break


# In[20]:


result.to_excel(u"C:/Users/n4663/Desktop/精灵相似匹配项目/result_csv.xlsx",encoding='utf-8')


# In[33]:





corpus_test1 = []
for i in data.loc[:,u"用户请求"]:
    corpus_test1.append(segment_jieba(i))


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
# yonghutiwen = u"哪些ssr值得培养"
# corpus_test1 = []
# corpus_test1.append(segment_jieba(yonghutiwen))
# corpus_test1 = []
# for i in data.loc[:,u"用户请求"]:
#     corpus_test1.append(segment_jieba(i))

# vec_test1 = vectorizer.transform(corpus)
# TF_test1 = transformer.transform(vec_test1)
# print len(list(TF_test1.toarray()[0]))
# max_match = {"ans":"","score":0,"train_ques":[]}
with codecs.open(u"C:/Users/n4663/Desktop/精灵相似匹配项目/match.txt",'w',encoding='utf-8') as w:
    for index,each_test in enumerate(tfidf_weight):
        max_similary = 0
        nearest_key = ""
        question = corpus[index]
        for each in label_center:
    #         print each
    #         print label_center[each]['nearest']
    #         print label_center[each]['center']
    #         print each_test
            degree = cosine_similarity([each_test,list(label_center[each]['center'])])[0][1]
            if degree > max_similary:
                max_similary =degree
                #print max_similary
                #print each
                nearest_key = label_center[each]['nearest']
                #print corpus[index]
        record = "\t".join([question,nearest_key,str(max_similary)]) + '\n'  
        w.write(record)


# print max_match["score"]
# print max_match["ans"]


# ### word2ver版本

# In[5]:





# In[14]:





# In[4]:


from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


# In[ ]:





# In[ ]:





# In[65]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import pandas as pd
jieba.load_userdict(u"C:/Users/n4663/Desktop/精灵相似匹配项目/yysword")
data = pd.read_csv(u"C:/Users/n4663/Desktop/精灵相似匹配项目/test.txt",sep="\t",encoding="utf-8")
#segment_jieba = lambda text: " ".join(jieba.cut(text))
corpus = []
for i in data.loc[:,u"用户请求"]:

    corpus.append(list(jieba.cut(i)))
model = Word2Vec(corpus, min_count=1,size=200,sg=0,seed=1,alpha=0.025,iter=5)

# model.save(u"C:/Users/n4663/Desktop/精灵相似匹配项目/word2vec_gensim")
# model = word2vec.Word2Vec.load(u"C:/Users/n4663/Desktop/精灵相似匹配项目/word2vec_gensim")
# model.train(more_sentences)


# In[110]:



ans_vec_ask = {}
for index,obj in enumerate(corpus):
    ans = data.loc[index,u"精灵答案"]
    ask = data.loc[index,u"用户请求"]
    ans_vec_ask.setdefault(ans,{"vec":[],"ask":[],"center":0,"nearest":"","max_score":0})
    sum_vec = np.zeros([1, 100])[0]
    for each in obj:
        sum_vec = model[each] + sum_vec
    ans_vec_ask[ans]["vec"].append(sum_vec)
    ans_vec_ask[ans]["ask"].append(ask)


# In[102]:


for i in ans_vec_ask:
    print ans_vec_ask[i]["ask"]
    break


# In[111]:


for i in ans_vec_ask:
    vecs =  ans_vec_ask[i]["vec"]
    
    y_pred = KMeans(n_clusters=1, random_state=9).fit(vecs)
    ans_vec_ask[i]["center"] = y_pred.cluster_centers_[0]
    max_ = 0
    for index,j in enumerate(ans_vec_ask[i]["vec"]):
        degree = cosine_similarity([ans_vec_ask[i]["center"],j])[0][1]
        #model.n_similarity(ans_vec_ask[i]["center"],j)
        if degree > max_:
            max_ = degree
            ans_vec_ask[i]["nearest"] = ans_vec_ask[i]["ask"][index]
            ans_vec_ask[i]["max_score"] = max_
    


# In[112]:




import codecs
import json
result =pd.DataFrame(columns=('ans','ask','nearest','max_score'))
for i in ans_vec_ask:
    ans = unicode(i)
    ask = unicode(json.dumps(ans_vec_ask[i]["ask"],ensure_ascii=False))
    nearest = unicode(ans_vec_ask[i]["nearest"])
    max_score = str('%.3f' % ans_vec_ask[i]["max_score"])
    result=result.append(pd.DataFrame({'ans':[ans],'ask':[ask],'nearest':[nearest],'max_score':[max_score]}),ignore_index=True)


import json
for i in ans_vec_ask:
    print i
    print json.dumps(ans_vec_ask[i]["nearest"],ensure_ascii=False)
    break


# In[113]:


result.to_excel(u"C:/Users/n4663/Desktop/精灵相似匹配项目/word2vecAVG_result_csv.xlsx",encoding='utf-8')


# ### word2vec 大话

# In[2]:


import codecs
data_dict = {}

with codecs.open(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/original_QA.txt",'r',encoding='utf-8') as f:
    for line in f:
        try:
            ques = line.strip().split('\t')[0]
            ans = line.strip().split('\t')[1]
            data_dict.setdefault(ans,[])
            data_dict[ans].append(ques)
        except:
            continue


# In[3]:


import json
with codecs.open(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/typeToAnswer.txt",'w',encoding='utf-8') as w:
    for index,each in enumerate(data_dict):
        ttype = "TYPE" + str(index)
        record = '\t'.join([ttype,json.dumps(data_dict[each],ensure_ascii=False)]) + "\n"
        w.write(record)
    #     print each
    #     break


# In[5]:


import json
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import pandas as pd
jieba.load_userdict(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/G18专属名词.txt")
data = pd.read_csv(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/typeToAnswer.txt",sep="\t",names=["type","ques"],encoding="utf-8")
#segment_jieba = lambda text: " ".join(jieba.cut(text))
corpus = []
for row in data.loc[:,"ques"]:
    #print json.loads(i)
    for each in json.loads(row):
        corpus.append(list(jieba.cut(each)))
# model = Word2Vec(corpus, min_count=1,size=200,sg=0,seed=1,alpha=0.025,iter=5)


# In[6]:


data.head(3)


# In[ ]:





# In[7]:


print json.dumps(corpus[0:3],ensure_ascii=False)


# In[8]:


print(len(corpus))


# In[ ]:





# In[9]:


get_ipython().run_cell_magic('time', '', 'model = Word2Vec(corpus, min_count=5,size=100,workers=4)')


# In[18]:





# In[11]:


# %%time
# more_sentences = corpus[10000:10010]
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences,total_examples=model.corpus_count,epochs=10000)


# In[10]:


model.save(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/word2vec_gensim")


# In[11]:


from gensim.models import word2vec
model = word2vec.Word2Vec.load(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/word2vec_gensim")


# In[ ]:





# In[12]:


import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
jieba.load_userdict(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/G18专属名词.txt")
ans_vec_ask = {}
for index,row in data.iterrows():
    #print(index)
    #print(list(row["ques"]))
    #print(json.loads(row["ques"]))
    ans = data.loc[index,u"type"]
    ans_vec_ask.setdefault(ans,{"vec":[],"ask":[],"center":0,"nearest":"","max_score":0,"dipin":[]})
    sum_vec = np.zeros([1, 100])[0]
    
    for asks in [(json.loads(row["ques"]))]:
        for each_ask in asks:
#             try:
#                 sentence_vec = np.zeros([1, 100])[0]
#                 for each_ask_vec in (list(jieba.cut(each_ask))):
#                     sentence_vec = model[each_ask_vec] + sentence_vec
# #                 print(each_ask)
# #                 print(sentence_vec)
#                 ans_vec_ask[ans]["vec"].append(sentence_vec/len(list(jieba.cut(each_ask))))
#                 ans_vec_ask[ans]["ask"].append(each_ask)
#             except:
#                 ans_vec_ask[ans]["dipin"].append(each_ask)
                
             ### 只跳过低频词
            sentence_vec = np.zeros([1, 100])[0]
            for each_ask_vec in (list(jieba.cut(each_ask))):
                try:
                    sentence_vec = model[each_ask_vec] + sentence_vec
                except:
                    continue
#                 print(each_ask)
#                 print(sentence_vec)
            ans_vec_ask[ans]["vec"].append(sentence_vec)
            ans_vec_ask[ans]["ask"].append(each_ask)
        
        
    #print(ans_vec_ask[ans])
    if len(ans_vec_ask[ans]["vec"]) > 0:
        y_pred = KMeans(n_clusters=1, random_state=9).fit(ans_vec_ask[ans]["vec"])
        ans_vec_ask[ans]["center"] = y_pred.cluster_centers_[0]
        #print(ans_vec_ask[ans]["center"])
        max_ = 0
        for index1,j in enumerate(ans_vec_ask[ans]["vec"]):
            #print(ans_vec_ask[ans]["center"])
            #print(j)
            #break
            degree = cosine_similarity([ans_vec_ask[ans]["center"],j])[0][1]
            #print(degree)
            #degree = model.n_similarity(ans_vec_ask[ans]["center"],j)
            if degree > max_:
                max_ = degree
                ans_vec_ask[ans]["nearest"] = ans_vec_ask[ans]["ask"][index1]
                ans_vec_ask[ans]["max_score"] = max_


# In[13]:


for i in ans_vec_ask:
    print(i)
    print(ans_vec_ask[i])
    break


# In[14]:


result =pd.DataFrame(columns=('ans','ask','nearest','max_score'))
for i in ans_vec_ask:
    ans = i
    ask = json.dumps(ans_vec_ask[i]["ask"],ensure_ascii=False)
    nearest = ans_vec_ask[i]["nearest"]
    max_score = str('%.3f' % ans_vec_ask[i]["max_score"])
    result=result.append(pd.DataFrame({'ans':[ans],'ask':[ask],'nearest':[nearest],'max_score':[max_score]}),ignore_index=True)


# In[15]:


ans_vec_ask["TYPE22796"]


# In[25]:


#result.to_excel(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/word2vec_dropword_result_csv.xlsx")
#result.to_csv(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/word2vec_dropword_result.txt",encoding='utf-8')


# In[24]:


import xlsxwriter
bb = pd.ExcelWriter(u"C:/Users/n4663/Desktop/精灵相似匹配项目/梦幻/word2vec_dropword_result_csv.xlsx",engine='xlsxwriter')
result.to_excel(bb, sheet_name='Sheet1')
bb.save()


# In[114]:


cosine_similarity([[0,0,0],[0,0,0]])[0][1]


# In[7]:


import jieba
import pandas as pd
jieba.load_userdict(u"C:/Users/n4663/Desktop/精灵相似匹配项目/大话/g17wordsv2.txt")
print(list(jieba.cut("恭喜你踏上了一条不归路")))


# In[ ]:




