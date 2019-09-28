# -*- coding: utf-8 -*-
# @Author: n4663
# @Date:   2019-08-07 18:44:31
# @Last Modified by:   n4663
# @Last Modified time: 2019-08-08 10:53:06
import requests
from datetime import datetime
from datetime import timedelta
import codecs
import pickle
import scoreOut
import json

def get_chat_data(date):
    chat_url = 'http://10.255.208.238:8080/static/yys/chat/{date}.log'.format(date=date)
    r = requests.get(chat_url)
    r.encoding = 'utf-8'
    data = {}
    for line in r.content.split('\n'):
        if u'发言时间'.encode('utf-8') not in line and len(line) > 10:
            chat_time,category,uid,urs,channel,roomId,channelId,isshieled,content,CCid = line.strip().split('\t')
            #print chat_time[0:-3], str(roomId),str(channelId),str(CCid)

            key = '^'.join([chat_time[0:-3], str(channelId), str(roomId), str(CCid)])
            data.setdefault(key, [])

            contentword = scoreOut.sent2word(content)
            contentvalus = scoreOut.sentiment_score_list(contentword)
            total_score = contentvalus[0][0]-contentvalus[0][1]

            data[key].append(total_score)

    return data

def get_filter(date):
    filter_set = set()
    for i in range(1,5):
        reward_url = 'http://10.255.208.238:8080/static/yys/9_37_{i}/{date}.log'.format(i=i,date=date)
        r = requests.get(reward_url)

        for line in r.content.split('\n'):
            try:
                json_data = json.loads(line.strip().split('[9_37_')[1][3:])
                if i == 1:

                    create_time = json_data['create_time'][0:-3]
                else:
                    create_time = json_data['time'][0:-3]
                sub_id = json_data['sub_id']
                key = '^'.join([create_time, str(sub_id)])
                filter_set.add(key)
            except Exception,e:
                continue

    return filter_set



if __name__ == '__main__':
    #CurrentDay = datetime.now().strftime('%Y%m%d')
    YesterDay = (datetime.now() + timedelta(days=-1)).strftime('%Y%m%d')
    data = get_chat_data(YesterDay)
    data_filter = get_filter(YesterDay)

    print len(data),len(data_filter)
    filename = YesterDay + '.txt'
    with codecs.open(filename,'w',encoding='utf-8') as w:
        title = '\t'.join(['time',u'房间id',u'频道id（房间唯一id）',u'CCid（主播id）',u'弹幕得分列表',u'总分',]) + '\n'
        w.write(title)
        for each in data:
            create_time,channelId,roomId,CCid = each.split('^')
            key = '^'.join([create_time,str(channelId)])
            if key not in data_filter:

                record = '\t'.join([create_time,str(roomId), str(channelId),str(CCid),str(data[each]),str(sum(list(data[each])))]) + '\n'
                w.write(record)








