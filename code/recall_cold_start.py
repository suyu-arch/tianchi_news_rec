import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
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
np.random.seed(2020)

# =========================
# 参数
# =========================
parser = argparse.ArgumentParser(description='冷启动召回（软过滤打分版）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

# embedding i2i 相似矩阵（建议使用你已有 emb_i2i_sim.pkl）
parser.add_argument('--emb_i2i_sim_path', default='./user_data2/sim/offline/emb_i2i_sim.pkl')

# item 元信息文件（至少包含 article_id, category_id, words_count, created_at_ts）
parser.add_argument('--item_info_path', default='./data/articles.csv')

# 候选与输出
parser.add_argument('--sim_item_topk', type=int, default=200, help='每个种子item取前N个相似item（粗召回）')
parser.add_argument('--raw_recall_num', type=int, default=300, help='每用户粗召回保留数量（软过滤前）')
parser.add_argument('--final_topk', type=int, default=50, help='每用户最终冷启动召回数量')

# 用户历史
parser.add_argument('--recent_n', type=int, default=3, help='用户最近N个点击作为种子')
parser.add_argument('--max_hist_len', type=int, default=50, help='构建用户画像时最多使用最近历史长度')

# 软过滤参数
parser.add_argument('--w_sim', type=float, default=1.0, help='embedding相似度权重')
parser.add_argument('--w_type', type=float, default=0.6, help='主题匹配奖励')
parser.add_argument('--w_words', type=float, default=0.4, help='字数接近奖励')
parser.add_argument('--w_time', type=float, default=0.5, help='时间接近奖励')
parser.add_argument('--penalty_seen', type=float, default=1.0, help='日志中出现过文章惩罚（冷启动核心）')

# 阈值（用于软惩罚，不是硬过滤）
parser.add_argument('--words_tol', type=float, default=250.0, help='字数容忍度')
parser.add_argument('--days_tol', type=float, default=120.0, help='发布时间容忍天数')

args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile

# 日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'cold_start recall mode={mode}')


def mms_one_user(scores):
    """归一化到[0,1]"""
    if len(scores) == 0:
        return scores
    mx, mn = max(scores), min(scores)
    if mx == mn:
        return [1e-3 for _ in scores]
    return [(s - mn) / (mx - mn) + 1e-3 for s in scores]


def load_item_info(item_info_path):
    """
    兼容常见列名：
      article_id / click_article_id
      category_id / category
      words_count / words
      created_at_ts / created_at
    """
    if item_info_path.endswith('.pkl'):
        item_info = pd.read_pickle(item_info_path)
    else:
        item_info = pd.read_csv(item_info_path)

    col_map = {}
    if 'article_id' in item_info.columns:
        col_map['article_id'] = 'article_id'
    elif 'click_article_id' in item_info.columns:
        col_map['article_id'] = 'click_article_id'
    else:
        raise ValueError('item_info缺少 article_id/click_article_id')

    if 'category_id' in item_info.columns:
        col_map['category'] = 'category_id'
    elif 'category' in item_info.columns:
        col_map['category'] = 'category'
    else:
        col_map['category'] = None

    if 'words_count' in item_info.columns:
        col_map['words'] = 'words_count'
    elif 'words' in item_info.columns:
        col_map['words'] = 'words'
    else:
        col_map['words'] = None

    if 'created_at_ts' in item_info.columns:
        col_map['created'] = 'created_at_ts'
    elif 'created_at' in item_info.columns:
        col_map['created'] = 'created_at'
    else:
        col_map['created'] = None

    item_info = item_info.copy()
    item_info['article_id'] = item_info[col_map['article_id']].astype(int)

    item_type_dict = {}
    item_words_dict = {}
    item_created_dict = {}

    if col_map['category'] is not None:
        item_type_dict = dict(zip(item_info['article_id'], item_info[col_map['category']]))

    if col_map['words'] is not None:
        item_words_dict = dict(zip(item_info['article_id'], item_info[col_map['words']].fillna(0).astype(float)))

    if col_map['created'] is not None:
        item_created_dict = dict(zip(item_info['article_id'], item_info[col_map['created']].fillna(0).astype(float)))

    return item_type_dict, item_words_dict, item_created_dict


def build_user_profile(df_click, item_type_dict, item_words_dict, item_created_dict, max_hist_len=50):
    """
    用户画像：
      - 历史主题集合
      - 历史平均字数
      - 最后一次点击文章创建时间
      - 用户点击序列
    """
    df_click = df_click.sort_values(['user_id', 'click_timestamp'])

    user_item = df_click.groupby('user_id')['click_article_id'].agg(list).to_dict()

    # 截断历史
    for u in user_item:
        if len(user_item[u]) > max_hist_len:
            user_item[u] = user_item[u][-max_hist_len:]

    user_type_set = {}
    user_mean_words = {}
    user_last_created = {}

    for u, items in user_item.items():
        types = []
        words = []
        createds = []

        for it in items:
            if it in item_type_dict:
                types.append(item_type_dict[it])
            if it in item_words_dict:
                words.append(item_words_dict[it])
            if it in item_created_dict:
                createds.append(item_created_dict[it])

        user_type_set[u] = set(types) if len(types) > 0 else set()
        user_mean_words[u] = float(np.mean(words)) if len(words) > 0 else 0.0
        user_last_created[u] = float(createds[-1]) if len(createds) > 0 else 0.0

    return user_item, user_type_set, user_mean_words, user_last_created


def coarse_recall_from_emb_i2i(user_id, user_item_dict, emb_i2i_sim, recent_n=3, sim_item_topk=200, raw_recall_num=300):
    """
    先用embedding i2i做粗召回，供冷启动过滤打分
    """
    if user_id not in user_item_dict:
        return []

    seeds = user_item_dict[user_id][::-1][:recent_n]
    rank = defaultdict(float)

    for loc, it in enumerate(seeds):
        loc_w = 0.7 ** loc
        sim_items = emb_i2i_sim.get(it, {})
        if not sim_items:
            continue

        top_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]
        for j, s in top_items:
            rank[j] += float(s) * loc_w

    rec = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:raw_recall_num]
    return rec


def soft_score_item(
    user_id, item_id, base_sim,
    user_type_set, user_mean_words, user_last_created,
    item_type_dict, item_words_dict, item_created_dict,
    click_article_ids_set,
    w_sim=1.0, w_type=0.6, w_words=0.4, w_time=0.5,
    penalty_seen=1.0, words_tol=250.0, days_tol=120.0
):
    score = w_sim * base_sim

    # 冷启动核心：日志中出现过则扣分（不是硬过滤）
    if item_id in click_article_ids_set:
        score -= penalty_seen

    # 主题匹配
    u_types = user_type_set.get(user_id, set())
    it_type = item_type_dict.get(item_id, None)
    if it_type is not None and len(u_types) > 0:
        if it_type in u_types:
            score += w_type
        else:
            score -= 0.2 * w_type

    # 字数接近
    u_words = user_mean_words.get(user_id, 0.0)
    it_words = item_words_dict.get(item_id, None)
    if it_words is not None and u_words > 0:
        diff = abs(it_words - u_words)
        score += w_words * np.exp(-diff / max(1.0, words_tol))

    # 发布时间接近（按天）
    u_last_ct = user_last_created.get(user_id, 0.0)
    it_ct = item_created_dict.get(item_id, None)
    if it_ct is not None and u_last_ct > 0:
        diff_days = abs(it_ct - u_last_ct) / (3600.0 * 24.0)
        score += w_time * np.exp(-diff_days / max(1.0, days_tol))

    return float(score)


@multitasking.task
def recall_worker(
    df_query_part, worker_id,
    user_item_dict, emb_i2i_sim,
    user_type_set, user_mean_words, user_last_created,
    item_type_dict, item_words_dict, item_created_dict, click_article_ids_set,
    recent_n, sim_item_topk, raw_recall_num, final_topk,
    w_sim, w_type, w_words, w_time, penalty_seen, words_tol, days_tol
):
    out = []

    for user_id, target_item in tqdm(df_query_part.values, desc=f'cold-worker-{worker_id}'):
        user_id = int(user_id)

        # 1) 粗召回
        raw_rec = coarse_recall_from_emb_i2i(
            user_id=user_id,
            user_item_dict=user_item_dict,
            emb_i2i_sim=emb_i2i_sim,
            recent_n=recent_n,
            sim_item_topk=sim_item_topk,
            raw_recall_num=raw_recall_num
        )
        if len(raw_rec) == 0:
            continue

        # 2) 软过滤打分
        rescored = []
        for item_id, base_sim in raw_rec:
            s = soft_score_item(
                user_id=user_id,
                item_id=int(item_id),
                base_sim=float(base_sim),
                user_type_set=user_type_set,
                user_mean_words=user_mean_words,
                user_last_created=user_last_created,
                item_type_dict=item_type_dict,
                item_words_dict=item_words_dict,
                item_created_dict=item_created_dict,
                click_article_ids_set=click_article_ids_set,
                w_sim=w_sim, w_type=w_type, w_words=w_words, w_time=w_time,
                penalty_seen=penalty_seen, words_tol=words_tol, days_tol=days_tol
            )
            rescored.append((int(item_id), s))

        # 用户内归一化 + 截断
        rescored = sorted(rescored, key=lambda x: x[1], reverse=True)
        item_ids = [x[0] for x in rescored[:final_topk]]
        scores = [x[1] for x in rescored[:final_topk]]
        scores = mms_one_user(scores)

        if len(item_ids) == 0:
            continue

        df_temp = pd.DataFrame({
            'user_id': user_id,
            'article_id': item_ids,
            'sim_score': scores
        })

        if target_item == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == int(target_item), 'label'] = 1

        out.append(df_temp[['user_id', 'article_id', 'sim_score', 'label']])

    if len(out) == 0:
        return

    df_out = pd.concat(out, ignore_index=True)
    os.makedirs('./user_data2/tmp/cold_start', exist_ok=True)
    df_out.to_pickle(f'./user_data2/tmp/cold_start/{worker_id}.pkl')


if __name__ == '__main__':
    # 数据路径
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')
        default_emb_path = './user_data2/sim/offline/emb_i2i_sim.pkl'
        save_file = './user_data2/data/offline/recall_cold_start.pkl'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')
        default_emb_path = './user_data2/sim/online/emb_i2i_sim.pkl'
        save_file = './user_data2/data/online/recall_cold_start.pkl'

    emb_i2i_sim_path = args.emb_i2i_sim_path
    if not os.path.exists(emb_i2i_sim_path):
        emb_i2i_sim_path = default_emb_path

    if not os.path.exists(emb_i2i_sim_path):
        raise FileNotFoundError(f'emb_i2i_sim 文件不存在: {emb_i2i_sim_path}')

    if not os.path.exists(args.item_info_path):
        raise FileNotFoundError(f'item_info 文件不存在: {args.item_info_path}')

    log.info(f'load emb sim: {emb_i2i_sim_path}')
    with open(emb_i2i_sim_path, 'rb') as f:
        emb_i2i_sim = pickle.load(f)

    # item特征
    item_type_dict, item_words_dict, item_created_dict = load_item_info(args.item_info_path)

    # 用户画像
    user_item_dict, user_type_set, user_mean_words, user_last_created = build_user_profile(
        df_click=df_click,
        item_type_dict=item_type_dict,
        item_words_dict=item_words_dict,
        item_created_dict=item_created_dict,
        max_hist_len=args.max_hist_len
    )

    # 日志中出现过的文章集合（冷启动目标：尽量推荐未出现文章）
    click_article_ids_set = set(df_click['click_article_id'].unique())

    # 清空临时目录
    os.makedirs('./user_data2/tmp/cold_start', exist_ok=True)
    for p, _, fs in os.walk('./user_data2/tmp/cold_start'):
        for fn in fs:
            os.remove(os.path.join(p, fn))

    # 并行分片
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_split = max_threads
    n_len = max(1, total // n_split)

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_part = df_query[df_query['user_id'].isin(part_users)][['user_id', 'click_article_id']]

        recall_worker(
            df_part, i,
            user_item_dict, emb_i2i_sim,
            user_type_set, user_mean_words, user_last_created,
            item_type_dict, item_words_dict, item_created_dict, click_article_ids_set,
            args.recent_n, args.sim_item_topk, args.raw_recall_num, args.final_topk,
            args.w_sim, args.w_type, args.w_words, args.w_time, args.penalty_seen, args.words_tol, args.days_tol
        )

    multitasking.wait_for_tasks()
    log.info('merge workers')

    # 合并
    df_data = pd.DataFrame()
    for p, _, fs in os.walk('./user_data2/tmp/cold_start'):
        for fn in fs:
            df_tmp = pd.read_pickle(os.path.join(p, fn))
            df_data = pd.concat([df_data, df_tmp], ignore_index=True)

    if len(df_data) > 0:
        df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_data.to_pickle(save_file)
    log.info(f'saved: {save_file}, shape={df_data.shape}')

    # 评估
    if mode == 'valid' and len(df_data) > 0:
        total_u = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        h5, m5, h10, m10, h20, m20, h40, m40, h50, m50 = evaluate(
            df_data[df_data['label'].notnull()], total_u
        )
        log.info(f'cold_start: h@5={h5:.6f}, h@10={h10:.6f}, h@20={h20:.6f}, h@40={h40:.6f}, h@50={h50:.6f}')