import argparse
import os
import pickle
import warnings
import gc  # 添加垃圾回收模块


import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import Logger,reduce_mem_usage

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pandarallel.initialize()

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args, unknown = parser.parse_known_args() 

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')

'''ItemCF特征：用户历史点击文章与待测文章的相似度加权和'''
def func_if_sum(x):#用户历史点击文章与待测文章的相似度加权和
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]#取用户最近点击的文章列表，按照时间顺序倒序排列，确保最近点击的文章在前面

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
            #item_sim是一个字典，存储了每篇文章与其他文章的相似度分数，item_sim[i][article_id]表示用户历史点击的第loc篇文章与待测文章的相似度分数，乘以一个衰减系数0.7的loc次方，表示越靠前（越近）的历史点击对相似度加权和的贡献越大
        except Exception as e:
            pass
    return sim_sum

'''ItemCF特征：用户历史点击的最后一篇文章与待测文章的相似度'''
def func_if_last(x):#用户历史点击的最后一篇文章与待测文章的相似度
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:#有时候last_time的相似度矩阵里面没有待测文章，这时候跳过
        pass
    return sim


if __name__ == '__main__':

    if mode == 'valid':
        df_feature = pd.read_pickle('./user_data2/data/offline/recall_itemCF.pkl')
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('./user_data2/data/online/recall_itemCF.pkl')
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
    # ========== 添加：加载后立即降内存 ==========
    print("优化 df_feature 内存...")
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    print("优化 df_click 内存...")
    df_click = reduce_mem_usage(df_click, verbose=True)
    # ===========================================

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv('./data/articles.csv')
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article,on='article_id', how='left')
    #将文章特征合并到找回结果的 DataFrame 中，按照文章ID进行合并，保留所有的行（how='left'），如果某些文章ID在 df_article 中没有对应的特征，则这些特征列会被填充为 NaN
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')#将文章的创建时间戳转换为 datetime 格式

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户行为统计特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)#按照用户ID和点击时间戳进行排序
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)#将点击数据中的 click_article_id 列重命名为 article_id，以便后续与文章特征进行合并
    df_click = df_click.merge(df_article, on='article_id', how='left')

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    #将用户点击的时间戳转换为 datetime 格式，errors='coerce' 参数表示如果无法解析某些时间戳（例如缺失值或格式错误），则将这些值设置为 NaT（Not a Time），而不是引发错误，这样可以保证后续的特征计算不会因为异常时间戳而中断
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour#提取用户点击时间的小时部分，作为一个新的特征列 click_datetime_hour，表示用户点击文章的时间段（0-23），这个特征可以帮助模型捕捉用户在不同时间段的行为模式和偏好

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        'user_id')['created_at_ts'].diff()#用户点击文章的创建时间差，按照用户ID分组，计算每次点击的文章创建时间与上一次点击的文章创建时间的差值，得到一个新的特征列 user_id_click_article_created_at_ts_diff，表示用户点击文章的创建时间差
    df_temp = df_click.groupby('user_id')['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')
    #将用户点击文章的创建时间差的平均值特征合并到召回结果的 DataFrame 中，按照用户ID进行合并，保留所有的行（how='left'），如果某些用户ID在 df_temp 中没有对应的特征，则这些特征列会被填充为 NaN

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby('user_id')['click_timestamp'].diff()
    df_temp = df_click.groupby('user_id')['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')#合并到召回结果里

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']#用户点击时间与文章创建时间的差值，表示用户点击文章时距文章创建的时间差，这个特征可以帮助模型捕捉用户对新旧文章的偏好，以及用户点击行为的时效性

    # 点击时间与文章创建时间差的平均值和标准差
    df_temp = df_click.groupby('user_id').agg(
    user_click_timestamp_created_at_ts_diff_mean=('click_timestamp_created_at_ts_diff', 'mean'),
    user_click_timestamp_created_at_ts_diff_std=('click_timestamp_created_at_ts_diff', 'std')).reset_index()

    df_feature = df_feature.merge(df_temp, on='user_id', how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')
    


    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby('user_id').agg(
        user_click_datetime_hour_std=('click_datetime_hour', 'std')
    ).reset_index()
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    '''文章特征'''
    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby('user_id').agg(
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', lambda x: x.iloc[-1])
    ).reset_index()
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_created_time=('created_at_ts', lambda x: x.iloc[-1]),
        user_clicked_article_created_time_max=('created_at_ts', 'max')
    ).reset_index()
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_click_time=('click_timestamp', lambda x: x.iloc[-1]),
        user_clicked_article_click_time_mean=('click_timestamp', 'mean')
    ).reset_index()#按照用户ID分组，计算每个用户点击的新闻的最后一次点击时间和平均点击时间，得到一个新的 DataFrame df_temp，其中包含用户ID、最后一次点击时间和平均点击时间等特征列
    df_feature = df_feature.merge(df_temp, on='user_id', how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    '''候选文章与用户行为的差值'''
    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']#用户点击的最后一篇文章的创建时间与待测文章创建时间的差值
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']#用户点击的最后一篇文章的点击时间与待测文章创建时间的差值
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']#用户点击的最后一篇文章的词数与待测文章词数的差值

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]#按照用户ID、文章ID、用户ID和类别ID分组，计算每个组的大小（即计数），得到一个新的 DataFrame df_temp，其中包含分组的特征列和对应的计数列，计数列的名称根据分组的特征列自动生成，例如 user_id_cnt、article_id_cnt、user_id_category_id_cnt 等

        df_feature = df_feature.merge(df_temp, on=f, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('./user_data2/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('./user_data2/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度(求和和最后一次)
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_sum, axis=1)#
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    # 降低内存占用，避免后续 read_pickle MemoryError
    # ========== 使用 reduce_mem_usage 替代 select_dtypes ==========
    print("开始最终内存优化...")
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    gc.collect()
    # =============================================================
    # for col in df_feature.select_dtypes(include=['int64']).columns:
    #     df_feature[col] = pd.to_numeric(df_feature[col], downcast='integer')
    # for col in df_feature.select_dtypes(include=['float64']).columns:
    #     df_feature[col] = pd.to_numeric(df_feature[col], downcast='float')

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('./user_data2/data/offline/feature_baseline.pkl')

    else:
        df_feature.to_pickle('./user_data2/data/online/feature_baseline.pkl')