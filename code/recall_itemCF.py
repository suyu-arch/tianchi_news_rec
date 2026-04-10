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
# 获取 CPU 核心数作为最大线程/进程数（multitasking 底层会适配）
max_threads = multitasking.config['CPU_CORES']
# 设置最大并发数
multitasking.set_max_threads(max_threads)
# 指定使用进程（process）而非线程（thread）作为执行引擎
multitasking.set_engine('process')
# 捕获 Ctrl+C 信号，触发 killall 终止所有正在运行的任务
signal.signal(signal.SIGINT, multitasking.killall)


random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args, unknown = parser.parse_known_args() 

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'itemcf 召回,mode: {mode}')


'''计算文章相似度，基于用户点击序列'''
def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()#按照原来的顺序，排过序的
    # 按用户分组，聚合点击的文章ID为列表,构建用户点击序列
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))
    # 将用户ID与其点击的文章列表构建成字典，方便后续计算相似度
    #user → item list

    item_cnt = defaultdict(int)# 统计每个文章被点击的次数
    sim_dict = {}
    # 存储文章相似度的字典，结构为 {item: {relate_item: sim_score}}

    for _, items in tqdm(user_item_dict.items()):#
        for loc1, item in enumerate(items):#
            item_cnt[item] += 1#
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)# 初始化相似度分数为0

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))#

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
# max_threads = multitasking.config['CPU_CORES']
# multitasking.set_max_threads(max_threads)
# multitasking.set_engine('process')
def recall(df_query, item_sim, user_item_dict, worker_id):#worker_id线程ID
    data_list = []

    for user_id, item_id in tqdm(df_query.values):# 遍历待预测列表中的每个用户及其对应的点击文章ID（验证集为真实点击，测试集为-1）
        rank = {}# 存储候选文章及其相似度分数，结构为 {relate_item: score}

        if user_id not in user_item_dict:
            continue # 如果用户没有点击记录，跳过该用户

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]# 取用户最近点击的2篇文章，进行相似文章召回
        #新闻具有时序性，用户最近点击的文章更能反映当前兴趣，因此取最近的2篇文章进行召回
        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:# 取相似度最高的200篇文章进行召回
                #key=lambda d: d[1] 表示按照相似度分数进行排序，reverse=True表示降序排序，[0:200]表示取前200个相似文章
                if relate_item not in interacted_items:#如果相似文章不在用户已点击的文章列表中，才进行召回
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)
                    # 根据相似度分数和位置权重计算最终的召回分数，距离已经点击的文章位置越近权重越高，最近点击文章权重是1，第二个点击权重是0.7
        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]# 取最终召回分数最高的100篇文章作为候选集
        item_ids = [item[0] for item in sim_items]# 提取候选文章ID列表
        item_sim_scores = [item[1] for item in sim_items]# 提取候选文章的相似度分数列表

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 构建标签：如果item_id为-1（测试集），标签设为NaN；否则标签为0，点击的文章标签为1
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0 #没有点击为负样本
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1#有点击

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('./user_data2/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'./user_data2/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')

        os.makedirs('./user_data2/sim/offline', exist_ok=True)
        sim_pkl_file = './user_data2/sim/offline/itemcf_sim.pkl'# 计算的相似度矩阵保存路径
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')

        os.makedirs('./user_data2/sim/online', exist_ok=True)
        sim_pkl_file = './user_data2/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)# 打乱用户顺序，避免同一批次都是相似用户，导致负载不均衡
    total = len(all_users)
    n_len = total // n_split# 每个线程处理的用户数量，最后一个线程可能会多处理一些用户（如果总用户数不能被线程数整除）

    # 清空临时文件夹
    for path, _, file_list in os.walk('./user_data2/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):# 按照每个线程处理的用户数量，分批次提交召回任务
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data2/tmp/itemcf'):# 遍历临时文件夹，读取每个线程的召回结果并合并到一个DataFrame中
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)# 按用户ID升序、相似度分数降序排序，保证每个用户的候选文章按照相似度从高到低排列
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        #df_query是验证集+测试集的待预测列表，df_query[df_query['click_article_id'] != -1] 取出验证集部分，user_id.nunique() 计算验证用户数量，作为召回指标的分母
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)#.notnull() 取出验证集部分（标签不为NaN），计算召回指标

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('./user_data2/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('./user_data2/data/online/recall_itemcf.pkl')
