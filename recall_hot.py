import argparse
import os
import random
import signal
import warnings
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

parser = argparse.ArgumentParser(description='hot 召回（Top类目保底 + 全局兜底）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--topk', type=int, default=50)
parser.add_argument('--beta', type=float, default=0.03, help='热度中“文章年龄”衰减系数(按小时)')
parser.add_argument('--user_cate_topn', type=int, default=3, help='每个用户取TopN偏好类目')
parser.add_argument('--per_cate_k', type=int, default=15, help='每个偏好类目保底召回数量')
parser.add_argument('--user_cate_decay', type=float, default=0.02, help='用户类目画像时间衰减系数(按小时)')
args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile
topk = args.topk
beta = args.beta
user_cate_topn = args.user_cate_topn
per_cate_k = args.per_cate_k
user_cate_decay = args.user_cate_decay

os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(
    f'hot recall, mode={mode}, topk={topk}, beta={beta}, user_cate_topn={user_cate_topn}, per_cate_k={per_cate_k}, user_cate_decay={user_cate_decay}'
)


def build_user_top_cates(df_click, df_article, topn=3, time_decay=0.02):
    article_cate = dict(zip(df_article['article_id'], df_article['category_id']))
    now_ts = df_click['click_timestamp'].max()

    df = df_click[['user_id', 'click_article_id', 'click_timestamp']].copy()
    df['category_id'] = df['click_article_id'].map(article_cate)
    df = df[df['category_id'].notnull()].copy()

    age_hours = (now_ts - df['click_timestamp']) / (1000 * 3600)
    age_hours = age_hours.clip(lower=0)
    df['w'] = np.exp(-time_decay * age_hours)

    g = df.groupby(['user_id', 'category_id'])['w'].sum().reset_index()

    user_top_cates = {}
    for user_id, sub in g.groupby('user_id'):
        sub = sub.sort_values('w', ascending=False).head(topn)
        user_top_cates[int(user_id)] = [int(x) for x in sub['category_id'].tolist()]
    return user_top_cates


def build_hot_pools(df_click, df_article, beta=0.03):
    article_ctime = dict(zip(df_article['article_id'], df_article['created_at_ts']))
    article_cate = dict(zip(df_article['article_id'], df_article['category_id']))
    now_ts = df_click['click_timestamp'].max()

    click_cnt = df_click.groupby('click_article_id').size().rename('click_cnt').reset_index()
    click_cnt.columns = ['article_id', 'click_cnt']
    click_cnt['created_at_ts'] = click_cnt['article_id'].map(article_ctime)
    click_cnt = click_cnt[click_cnt['created_at_ts'].notnull()].copy()

    click_cnt['age_hours'] = (now_ts - click_cnt['created_at_ts']) / (1000 * 3600)
    click_cnt['age_hours'] = click_cnt['age_hours'].clip(lower=0)

    # 热度分：点击量 + 年龄衰减
    click_cnt['hot_score'] = np.log1p(click_cnt['click_cnt']) * np.exp(-beta * click_cnt['age_hours'])
    click_cnt['category_id'] = click_cnt['article_id'].map(article_cate)

    hot_global = click_cnt.sort_values('hot_score', ascending=False)[['article_id', 'hot_score']].values.tolist()

    hot_by_cate = {}
    for cate, g in click_cnt.groupby('category_id'):
        hot_by_cate[int(cate)] = g.sort_values('hot_score', ascending=False)[['article_id', 'hot_score']].values.tolist()

    return hot_global, hot_by_cate


def pick_candidates_hot(user_id, clicked_set, user_top_cates, hot_by_cate, hot_global,
                        topk=50, per_cate_k=15):
    rec = []
    used = set()

    cates = user_top_cates.get(user_id, [])
    for cate in cates:
        if cate not in hot_by_cate:
            continue
        cnt = 0
        for aid, score in hot_by_cate[cate]:
            aid = int(aid)
            if aid in clicked_set or aid in used:
                continue
            rec.append((aid, float(score)))
            used.add(aid)
            cnt += 1
            if cnt >= per_cate_k:
                break

    if len(rec) < topk:
        for aid, score in hot_global:
            aid = int(aid)
            if aid in clicked_set or aid in used:
                continue
            rec.append((aid, float(score)))
            used.add(aid)
            if len(rec) >= topk:
                break

    return rec[:topk]


@multitasking.task
def recall(df_query, user_item_dict, user_top_cates, hot_global, hot_by_cate,
           worker_id, topk, per_cate_k):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        clicked_set = user_item_dict.get(user_id, set())

        rec = pick_candidates_hot(
            user_id=int(user_id),
            clicked_set=clicked_set,
            user_top_cates=user_top_cates,
            hot_by_cate=hot_by_cate,
            hot_global=hot_global,
            topk=topk,
            per_cate_k=per_cate_k
        )

        if len(rec) == 0:
            continue

        item_ids = [x[0] for x in rec]
        sim_scores = [x[1] for x in rec]

        df_temp = pd.DataFrame({
            'user_id': [int(user_id)] * len(item_ids),
            'article_id': item_ids,
            'sim_score': sim_scores
        })

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == int(item_id), 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        data_list.append(df_temp)

    if len(data_list) == 0:
        return

    df_data = pd.concat(data_list, sort=False)
    os.makedirs('./user_data2/tmp/hot', exist_ok=True)
    df_data.to_pickle(f'./user_data2/tmp/hot/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')
        df_article = pd.read_csv('./data/articles.csv')
        save_file = './user_data2/data/offline/recall_hot.pkl'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')
        df_article = pd.read_csv('./data/articles.csv')
        save_file = './user_data2/data/online/recall_hot.pkl'

    log.debug(f'df_click shape: {df_click.shape}, df_query shape: {df_query.shape}, df_article shape: {df_article.shape}')

    # 用户点击集合
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    # 用户Top类目画像
    user_top_cates = build_user_top_cates(df_click, df_article, topn=user_cate_topn, time_decay=user_cate_decay)

    # 热度池
    hot_global, hot_by_cate = build_hot_pools(df_click, df_article, beta=beta)

    # 并行召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = max(1, total // n_split)

    os.makedirs('./user_data2/tmp/hot', exist_ok=True)
    for path, _, file_list in os.walk('./user_data2/tmp/hot'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)][['user_id', 'click_article_id']]
        recall(df_temp, user_item_dict, user_top_cates, hot_global, hot_by_cate, i, topk, per_cate_k)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data2/tmp/hot'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    if mode == 'valid':
        total_u = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total_u
        )
        log.debug(f'hot: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}')

    df_data.to_pickle(save_file)