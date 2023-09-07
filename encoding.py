# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:02:29 2023

@author: c
"""
from jina import requests,Executor
from clip_client import Client
import numpy as np
import pandas as pd
from Graph_processing import data_processing
import os
import asyncio
import warnings
warnings.filterwarnings('ignore')
# or pip3 install tornado==4.5.3
import nest_asyncio
nest_asyncio.apply()

# 
import sys
import traceback
from asyncio import CancelledError  # inputs is not valid!
from jina.excepts import BadClient,BadServerFlow  # url orkey is error
import time

class bert_encoding(Executor):
    
    def __init__(self,
                 url:str = 'grpcs://api.clip.jina.ai:2096',
                 key:str = None,
                 batch_size:int = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.url = url
        self.key = key
        self.encoder = Client(url,credential={'Authorization': key},batch_size = batch_size)
        self.batch_size = batch_size
        self.error = 0
        
        
    @requests(on='/')
    async def encoding(self,x,**kwargs):
        
        while True:
        
            try:
                t = await self.encoder.aencode(content=x)
                self.error = 0
                break
            
            except ConnectionError:
                self.encoder = Client(self.url,credential={'Authorization': self.key},batch_size = self.batch_size)
                time.sleep(1)
                
                continue
            
            except BadServerFlow:
                time.sleep(60*5)
                continue
            
            except CancelledError as e:
                
                if self.error < 10:
                    self.encoder = Client(self.url,credential={'Authorization': self.key},batch_size = self.batch_size)
                    time.sleep(3)
                    self.error += 1
                    
                    continue
                
                else:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=open('Error_Infos.txt','w+'))
                    break
            
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=open('Error_Infos.txt','w+'))
                break
            
        return t


import spacy
class spacy_encoding:
    
    def __init__(self,
                 model_name:str = 'zh_core_web_md'):
        
        # zh_core_web_lg 575MB
        
        self.nlp = spacy.load(model_name)
        
    def enconding(self,x):
        
        return x.map(lambda x: self.nlp(x).vector)
        
        
        
if __name__ == '__main__':
    raw_data_path = r'.\rumdect'
    label_raw = os.path.join(raw_data_path,'Weibo.txt')
    news_path = os.path.join(raw_data_path,'Weibo')
    
    label = pd.read_csv(label_raw,sep=":|\t",names=[],engine='python')
    label = label.loc[:,[1,3]].rename({1:'id',3:'label'}, axis='columns')
    news_files = os.listdir(news_path)
    
    
    DP = data_processing(news_path)
    DP.graph_processing(news_files[0])
    
    bert_encoder = bert_encoding(key='<your token key>')
    
    t = asyncio.run(bert_encoder.encoding(DP.graph['x']['text'].to_list()))
    
    
    
