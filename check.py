# -*- coding: utf-8 -*-
"""
Created on Sat May 27 16:15:11 2023

@author: c
"""
import os
import pandas as pd
import numpy as np
import asyncio

from Graph_processing import data_processing

from encoding import bert_encoding,spacy_encoding


raw_data_path = r'./rumdect'
spacy_model_name = 'zh_core_web_lg'   # https://spacy.io/models/zh#zh_core_web_lg
bert_key = '<your token key>'         # https://cloud.jina.ai/user/inference-api
# raw data
label_raw = os.path.join(raw_data_path,'Weibo.txt')
news_path = os.path.join(raw_data_path,'Weibo')

news_files = os.listdir(news_path)
npy_files = os.listdir(os.path.join(raw_data_path, 'embeddings','spacy'))


label = pd.read_csv(label_raw,sep=":|\t",names=[],engine='python')
label = label.loc[:,[1,3]].rename({1:'id',3:'label'}, axis='columns')


# graph_processer & encoders
DP = data_processing(news_path,label=label)

# bert_encoder = bert_encoding(key=bert_key)
spacy_encoder = spacy_encoding(model_name=spacy_model_name)

for i in range(len(news_files)):
    
    if news_files[i][:-5] + '.npy' not in npy_files:
        
        print(f'\rProcessing: {i+1}/{len(news_files)}',end='')
        DP.graph_processing(news_files[i])
    

        graph = DP.graph

        # bert encoding
        # t = asyncio.run(bert_encoder.encoding(graph['x']['text'].to_list()))    # t -> (768)
    
        # x = np.array([np.concatenate((t,graph['x'].iloc[:,5:].values),axis=-1).astype('float32'),
        #           np.array(graph['edge_index']),graph['y'].values])
        
        # np.save( os.path.join(raw_data_path,'embeddings','bert',f'{news_files[i][:-5]}'), x)
        
        
        # spacy encoding
        t = spacy_encoder.enconding(graph['x']['text'])
        t = np.vstack(tuple([i for i in t]))
        x = np.array([np.concatenate((t,graph['x'].iloc[:,5:].values),axis=-1).astype('float32'),
                      np.array(graph['edge_index']),graph['y'].values])
        np.save( os.path.join('./data/Weibo/raw/spacy',f'{news_files[i][:-5]}'), x)
        
        # profile encoding
        # x = np.array([graph['x'].iloc[:,5:].values.astype('float32'),
        #               np.array(graph['edge_index']),graph['y'].values])
        
        # np.save( os.path.join(raw_data_path,'embeddings','profile',f'{news_files[i][:-5]}'), x)
    
    
    
    
    