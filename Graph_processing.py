# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:15:39 2023

@author: c
"""
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings('ignore')

class data_processing:
    
    def __init__(self,data_path = None,label=None):
        self.data_path = data_path
        self.label = label
        
        self.profile_feature = ['uid','screen_name',  'text','original_text', 't',
                       'city', 'province',
                       'friends_count',  'reposts_count' , 'bi_followers_count',
                       'attitudes_count', 'followers_count', 'verified_type', 
                       'statuses_count', 'verified','favourites_count', 'gender',
                       'user_geo_enabled', 'comments_count']
        
        self.user = pd.DataFrame(columns=self.profile_feature)
        
        self.user_id =[]
        self.count_per_graph = []
        self.reset_graph()
        
        
    @staticmethod
    def json2df(json_path):
    
        with open(json_path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return pd.json_normalize(data)
    
    def user_processing(self,json_name):
        
        df = self.json2df(os.path.join(self.data_path,json_name))
        
        for i,uid in enumerate(df['uid']):

            if uid not in self.user['uid'].values:
                
                self.user = pd.concat([self.user,df.loc[i,:].to_frame().T],join='inner',ignore_index=True)

            else:
                self.user.loc[self.user['uid']==uid,'text'] += df.loc[i,'text']
    
    
    def reset_graph(self):
        
        self.graph = {'x':[],
                      'edge_index':[],
                      'y':[]}
        
    def user_count(self,json_name):
        df = self.json2df(os.path.join(self.data_path,json_name))
        
        self.count_per_graph.append(df.shape[0])
        
        for _,uid in enumerate(df['uid']):
            if uid not in self.user_id:
                self.user_id.append(uid)
                
    def save(self,path,save=True):
        if save:
            self.user.to_csv(path,encoding='utf-8')
        else:
            self.user = pd.read_csv(path)
            
    def graph_processing(self,json_name,label=None):
        
 
        df = self.json2df(os.path.join(self.data_path,json_name))
        
        df = df.loc[:,self.profile_feature]
        df['verified'] = df['verified'].map({True:1,False:0})
        df['user_geo_enabled'] = df['user_geo_enabled'].map({True:1,False:0})
        df['gender'] = df['gender'].map({'m':1,'f':0})
        
        
        idx = df[(df['text'] == '') | (df['text']=='转发微博')].index
        df = df.drop(index = idx).reset_index(drop=True)
        
        
        edge_index = [[],[]]    # [[src],[des]]
        
        for i in range(len(df)):
            if '//@' in df.loc[i,'original_text']:
                
                tweet_name = df.loc[i,'original_text'].split('//@')[1].split(':')[0]
                tweet = df[df['screen_name']==tweet_name]
                if len(tweet)==1:
                    j = tweet.index[0]
                
                    edge_index[0].append(j)
                    edge_index[1].append(i)
                else:
                    edge_index[0].append(0)
                    edge_index[1].append(i)
            else:
                edge_index[0].append(0)
                edge_index[1].append(i)
        
        self.graph['x'] = df
        self.graph['edge_index'] = edge_index
        
        
        self.graph['y'] = self.label[self.label['id']==int(json_name.split('.')[0])]['label'].values
        
        return 
    
    
    
if __name__ == '__main__':
    raw_data_path = r'.\rumdect'
    label_raw = os.path.join(raw_data_path,'Weibo.txt')
    news_path = os.path.join(raw_data_path,'Weibo')
    
    label = pd.read_csv(label_raw,sep=":|\t",names=[],engine='python')
    label = label.loc[:,[1,3]].rename({1:'id',3:'label'}, axis='columns')
    news_files = os.listdir(news_path)
    DP = data_processing(news_path)
    DP.graph_processing(news_files[0])
    