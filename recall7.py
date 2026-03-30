import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate


warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--target_recall_num', type=int, default=150, help='每用户最终召回数量')

args, unknown = parser.parse_known_args() 

mode = args.mode
logfile = args.logfile
target_recall_num = args.target_recall_num

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'召回合并: {mode}')
log.info(f'召回合并: {mode}, target_recall_num={target_recall_num}')

'''Min-Max 归一化函数'''
def mms(df):
    """向量化 Min-Max 标准化"""
    df = df.sort_values(['user_id', 'sim_score'], ascending=[True, False]).copy()
    g = df.groupby('user_id')['sim_score']
    smax = g.transform('max')
    smin = g.transform('min')
    denom = (smax - smin).replace(0, np.nan)
    df['sim_score'] = ((df['sim_score'] - smin) / denom).fillna(1.0) + 1e-3
    return df['sim_score'].values

'''计算召回结果的相似度'''
def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]
        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]
            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')
        recall_path = './user_data2/data/offline'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')
        recall_path = './user_data2/data/online'

    log.debug(f'max_threads {max_threads}')

    # 召回方法列表和对应权重
    recall_methods = ['itemcf', 'w2v', 'binetwork', 'youtubednn', 'swing', 'cold_start']
    weights = {
        'itemcf': 1, 
        'binetwork': 1, 
        'w2v': 0.2,
        'youtubednn': 0.01,
        'swing': 0.03,
        'cold_start': 0.06,
    }
    log.info(f'召回方法: {recall_methods}')
    log.info(f'权重: {weights}')

    recall_list = []
    recall_dict = {}
    
    for recall_method in recall_methods:
        recall_file = f'{recall_path}/recall_{recall_method}.pkl'
        recall_result = pd.read_pickle(recall_file)
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result
        log.info(f'加载 {recall_method}: shape={recall_result.shape}, weight={weight}')

    # # 求相似度
    # if len(recall_methods) >= 2:
    #     log.info('='*60)
    #     log.info('召回方法相似度分析:')
    #     for recall_method1, recall_method2 in permutations(recall_methods, 2):
    #         score = recall_result_sim(recall_dict[recall_method1],
    #                                   recall_dict[recall_method2])
    #         log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score:.4f}')

    # 合并召回结果
    log.info('='*60)
    log.info('合并召回结果...')
    
    recall_final = pd.concat(recall_list, sort=False)
    
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()#

    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    
    recall_final = recall_final.merge(recall_score, how='left', 
                                      on=['user_id', 'article_id'])

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])
    #做前target_recall_num的截断，后续评估和分析都基于这个截断后的结果
    recall_final = recall_final.groupby('user_id').head(target_recall_num).reset_index(drop=True)

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final head:\n{recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg, desc='过滤无正样本用户'):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                log.warning(f'用户 {user_id} 有多个正样本，可能存在数据问题')
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    
    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    # 计算评估指标
    if mode == 'valid':
        log.info('='*60)
        log.info('评估指标:')
        
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        
        df_eval = df_useful_recall[df_useful_recall['label'].notnull()]
        
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, \
        hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(df_eval, total)

        metrics_log = (
            f'HitRate@5={hitrate_5:.6f}, MRR@5={mrr_5:.6f}\n'
            f'HitRate@10={hitrate_10:.6f}, MRR@10={mrr_10:.6f}\n'
            f'HitRate@20={hitrate_20:.6f}, MRR@20={mrr_20:.6f}\n'
            f'HitRate@40={hitrate_40:.6f}, MRR@40={mrr_40:.6f}\n'
            f'HitRate@50={hitrate_50:.6f}, MRR@50={mrr_50:.6f}'
        )
        log.info(f'召回合并后指标:\n{metrics_log}')

    # 统计召回数量分布
    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    
    min_recall = df['cnt'].min()
    max_recall = df['cnt'].max()
    mean_recall = df['cnt'].mean()
    median_recall = df['cnt'].median()
    
    log.info('='*60)
    log.info('召回数量统计:')
    log.info(f"最少召回: {min_recall}")
    log.info(f"最多召回: {max_recall}")
    log.info(f"平均召回: {mean_recall:.2f}")
    log.info(f"中位召回: {median_recall:.2f}")
    
    # 分位数统计
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    log.info(f"分位数分布: ")
    for p in percentiles:
        val = np.percentile(df['cnt'], p)
        log.info(f"  P{p}: {val:.0f}")

    # 标签分布统计
    labeled_data = df_useful_recall[df_useful_recall['label'].notnull()]
    if len(labeled_data) > 0:
        label_dist = labeled_data['label'].value_counts()
        log.info(f"标签分布:\n{label_dist}")

    # 保存到本地
    if mode == 'valid':
        output_file = './user_data2/data/offline/recall7.pkl'
    else:
        output_file = './user_data2/data/online/recall7.pkl'
    
    df_useful_recall.to_pickle(output_file)
    log.info(f'召回结果已保存: {output_file}')
    log.info('='*60)