import argparse
import os
import pickle
import random
import math
import signal
import warnings
from collections import defaultdict
from itertools import combinations
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
parser = argparse.ArgumentParser(description='swing 召回（加速版）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

# ===== 新增强可控参数 =====
parser.add_argument('--alpha', type=float, default=1.0, help='swing分母平滑项')
parser.add_argument('--topk', type=int, default=50, help='每用户最终召回条数')
parser.add_argument('--recent_n', type=int, default=2, help='最近N篇作为种子')
parser.add_argument('--topn_per_seed', type=int, default=80, help='每个种子item取前N个相似item')
parser.add_argument('--max_users_per_item', type=int, default=80, help='每个item最多采样多少用户做用户对')
parser.add_argument('--max_items_per_user', type=int, default=200, help='每个用户最多保留最近多少点击用于建图')
parser.add_argument('--max_common_items', type=int, default=80, help='用户对的共同item最多保留多少个')
parser.add_argument('--rebuild_sim', type=int, default=0, help='1强制重建sim，0优先读取缓存')

args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile

os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(
    f'swing 召回(加速), mode={mode}, alpha={args.alpha}, topk={args.topk}, recent_n={args.recent_n}, '
    f'topn_per_seed={args.topn_per_seed}, max_users_per_item={args.max_users_per_item}, '
    f'max_items_per_user={args.max_items_per_user}, max_common_items={args.max_common_items}, rebuild_sim={args.rebuild_sim}'
)


def _trim_user_items(user_item_dict, max_items_per_user=200):
    """截断用户历史长度，降低交集计算成本"""
    new_dict = {}
    for u, items in user_item_dict.items():
        if len(items) > max_items_per_user:
            new_dict[u] = items[-max_items_per_user:]
        else:
            new_dict[u] = items
    return new_dict


'''基于用户点击行为构建文章之间的相似度矩阵（加速版）'''
def cal_sim(
    df,
    alpha=1.0,
    max_users_per_item=80,#每个item最多采样多少用户做用户对，热门item用户过多时采样部分用户，降低计算成本
    max_items_per_user=200,#每个用户最多保留最近多少点击用于建图，用户历史过长时截断，降低计算成本
    max_common_items=80#用户对的共同item最多保留多少个，用户对的共同item过多时截断，降低计算成本
):
    df = df.sort_values(['user_id', 'click_timestamp'])

    user_item_ = df.groupby('user_id')['click_article_id'].agg(list).reset_index()#每个用户的点击文章列表
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    user_item_dict = _trim_user_items(user_item_dict, max_items_per_user=max_items_per_user)#截断用户历史长度，降低交集计算成本

    item_user_ = df.groupby('click_article_id')['user_id'].agg(list).reset_index()#每篇文章的点击用户列表
    item_user_dict = dict(zip(item_user_['click_article_id'], item_user_['user_id']))

    # 集合用于交集加速
    user_item_set = {u: set(v) for u, v in user_item_dict.items()}

    # 用户活跃度长度与用户权重
    user_len = {u: len(items) for u, items in user_item_set.items()}#每个用户的历史点击文章数量
    user_norm_w = {
        u: (1.0 / math.sqrt(l)) if l > 0 else 0.0
        for u, l in user_len.items()
    }

    sim_dict = {}
    pair_weight = defaultdict(float)

    # 1) 构建用户对权重（热门item限采样）
    for item, users in tqdm(item_user_dict.items(), desc='build user-pair weight'):
        users = list(set(users))
        # 过滤掉不在截断后user_item_dict中的用户（极端情况保护）
        users = [u for u in users if u in user_item_set]

        if len(users) < 2:
            continue

        if len(users) > max_users_per_item:
            users = random.sample(users, max_users_per_item)#热门item用户过多时采样部分用户，降低计算成本

        for u, v in combinations(users, 2):#
            inter = len(user_item_set[u] & user_item_set[v])#用户对的共同item数量
            w = user_norm_w[u] * user_norm_w[v] / (alpha + inter)
            pair_weight[(u, v)] += w
            pair_weight[(v, u)] += w

    # 2) 用户对权重 -> 物品对
    for (u, v), w_uv in tqdm(pair_weight.items(), desc='push weight to item-item'):
        common_items = list(user_item_set[u] & user_item_set[v])#用户对的共同item列表
        if len(common_items) < 2:
            continue

        if len(common_items) > max_common_items:#用户对的共同item过多时截断，降低计算成本
            common_items = common_items[:max_common_items]

        for i, j in combinations(common_items, 2):
            sim_dict.setdefault(i, {})
            sim_dict.setdefault(j, {})
            sim_dict[i][j] = sim_dict[i].get(j, 0.0) + w_uv
            sim_dict[j][i] = sim_dict[j].get(i, 0.0) + w_uv

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, swing_sim, user_item_dict, worker_id, recent_n=2, topn_per_seed=80, topk=50):
    data_list = []

    for user_id, item_id in tqdm(df_query.values, desc=f'recall-{worker_id}'):
        if user_id not in user_item_dict:
            continue

        rank = {}#最终得分 = 各种子item相似度*位置权重之和
        interacted_items = user_item_dict[user_id]
        seeds = interacted_items[::-1][:recent_n]#最近N篇作为种子
        seed_set = set(seeds)

        # 每个种子取topn_per_seed
        for loc, item in enumerate(seeds):
            loc_weight = 0.7 ** loc
            sim_items = swing_sim.get(item, {})
            if not sim_items:
                continue

            # 比全量sorted更省：先转list再截断
            top_items = sorted(sim_items.items(), key=lambda d: d[1], reverse=True)[:topn_per_seed]#每个种子item取前N个相似item
            for relate_item, wij in top_items:
                if relate_item in seed_set:
                    continue
                rank[relate_item] = rank.get(relate_item, 0.0) + wij * loc_weight

        # 最终截断topk
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:topk]
        if len(sim_items) == 0:
            continue

        item_ids = [x[0] for x in sim_items]
        item_sim_scores = [x[1] for x in sim_items]

        df_temp = pd.DataFrame({
            'user_id': int(user_id),
            'article_id': item_ids,
            'sim_score': item_sim_scores
        })

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == int(item_id), 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    if len(data_list) == 0:
        return

    df_data = pd.concat(data_list, sort=False)
    os.makedirs('./user_data2/tmp/swing', exist_ok=True)
    df_data.to_pickle(f'./user_data2/tmp/swing/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')

        os.makedirs('./user_data2/sim/offline', exist_ok=True)
        sim_pkl_file = './user_data2/sim/offline/swing_sim.pkl'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')

        os.makedirs('./user_data2/sim/online', exist_ok=True)
        sim_pkl_file = './user_data2/sim/online/swing_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'df_query shape: {df_query.shape}')

    # ===== sim缓存逻辑：默认优先加载 =====
    if os.path.exists(sim_pkl_file) and args.rebuild_sim == 0:
        log.info(f'加载已缓存的swing_sim: {sim_pkl_file}')
        with open(sim_pkl_file, 'rb') as f:
            swing_sim = pickle.load(f)

        # user_item_dict仍需按当前点击数据重建（用于用户种子）
        user_item_ = df_click.sort_values(['user_id', 'click_timestamp']).groupby('user_id')['click_article_id'].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
        user_item_dict = _trim_user_items(user_item_dict, max_items_per_user=args.max_items_per_user)
    else:
        log.info('开始重建swing_sim...')
        swing_sim, user_item_dict = cal_sim(
            df_click,
            alpha=args.alpha,
            max_users_per_item=args.max_users_per_item,
            max_items_per_user=args.max_items_per_user,
            max_common_items=args.max_common_items
        )
        with open(sim_pkl_file, 'wb') as f:
            pickle.dump(swing_sim, f)
        log.info(f'swing_sim已保存: {sim_pkl_file}')

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)#用户打乱后分批，避免同一批用户过于集中，导致某些批次计算过慢（如热门用户过多时）
    total = len(all_users)
    n_len = max(1, total // n_split)  # 修复：防止为0

    os.makedirs('./user_data2/tmp/swing', exist_ok=True)
    for path, _, file_list in os.walk('./user_data2/tmp/swing'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]#每批用户的数量
        df_temp = df_query[df_query['user_id'].isin(part_users)][['user_id', 'click_article_id']]
        recall(
            df_temp, swing_sim, user_item_dict, i,
            recent_n=args.recent_n,
            topn_per_seed=args.topn_per_seed,
            topk=args.topk
        )

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data2/tmp/swing'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    if len(df_data) == 0:
        log.info('召回结果为空，请调整参数（可增大topn_per_seed/topk）')
    else:
        df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
        log.debug(f'df_data.head:\n{df_data.head()}')
        log.debug(f'df_data.shape: {df_data.shape}')

    # 计算召回指标
    if mode == 'valid' and len(df_data) > 0:
        log.info('计算召回指标')
        total_eval = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total_eval
        )

        log.debug(
            f'swing: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('./user_data2/data/offline/recall_swing.pkl')
        log.info('已保存 ./user_data2/data/offline/recall_swing.pkl')
    else:
        df_data.to_pickle('./user_data2/data/online/recall_swing.pkl')
        log.info('已保存 ./user_data2/data/online/recall_swing.pkl')