import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='binetwork 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args, unknown = parser.parse_known_args() 

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'binetwork 召回，mode: {mode}')


'''基于用户点击行为构建文章之间的相似度矩阵。'''
def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        list).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        list).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))
    #按照文章ID分组，聚合点击该文章的用户ID为列表，构建文章被哪些用户点击过的字典

    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):# 遍历每篇文章及其对应的点击用户列表
        sim_dict.setdefault(item, {})

        for user in users:
            tmp_len = len(user_item_dict[user])# 获取该用户点击的文章数量，作为权重计算的一部分，点击越多的用户对相似度的贡献越小（通过对数函数进行平滑处理）
            for relate_item in user_item_dict[user]:# 遍历该用户点击的每篇文章，计算与当前文章的相似度贡献
                # 跳过自身（或可选：设为1）
                if item == relate_item:
                    continue
                sim_dict[item].setdefault(relate_item, 0)
                sim_dict[item][relate_item] += 1 / \
                    (math.log(len(users)+1) * math.log(tmp_len+1))
                # 计算两篇文章的相似度分数，分子是共同点击该文章的用户数量，分母是该文章被点击的用户数量和用户点击的文章数量的对数乘积，平滑处理避免极端值影响相似度计算
            
    return sim_dict, user_item_dict

@multitasking.task
# max_threads = multitasking.config['CPU_CORES']
# multitasking.set_max_threads(max_threads)
# multitasking.set_engine('process')

def recall(df_query, binetwork_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:1]
        # 取用户最近点击的1篇文章，进行相似文章召回，binetwork只取最近点击的1篇文章进行召回，进一步突出时序性，减少噪声

        for _, item in enumerate(interacted_items):
            for relate_item, wij in sorted(binetwork_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:100]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids# 将候选文章ID列表赋值给DataFrame的'article_id'列
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1#

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('./user_data2/tmp/binetwork', exist_ok=True)
    df_data.to_pickle(f'./user_data2/tmp/binetwork/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')

        os.makedirs('./user_data2/sim/offline', exist_ok=True)
        sim_pkl_file = './user_data2/sim/offline/binetwork_sim.pkl'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')

        os.makedirs('./user_data2/sim/online', exist_ok=True)
        sim_pkl_file = './user_data2/sim/online/binetwork_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    binetwork_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(binetwork_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('./user_data2/tmp/binetwork'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, binetwork_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data2/tmp/binetwork'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'binetwork: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('./user_data2/data/offline/recall_binetwork.pkl')
    else:
        df_data.to_pickle('./user_data2/data/online/recall_binetwork.pkl')
