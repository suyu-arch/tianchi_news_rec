import argparse
import gc
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from utils import Logger, reduce_mem_usage

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

seed = 2020
np.random.seed(seed)


parser = argparse.ArgumentParser(description='排序特征 v3（仅保留简历主用特征）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--max_user_candidates', type=int, default=0)
parser.add_argument('--max_history_for_sim_sum', type=int, default=20)
parser.add_argument('--pair_chunk_size', type=int, default=500000)
args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile
max_user_candidates = max(0, args.max_user_candidates)
max_history_for_sim_sum = max(1, args.max_history_for_sim_sum)
pair_chunk_size = max(100000, args.pair_chunk_size)

os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(
    'rank_feature3 (selected features only), mode=%s, max_user_candidates=%s, max_history_for_sim_sum=%s, pair_chunk_size=%s',
    mode,
    max_user_candidates,
    max_history_for_sim_sum,
    pair_chunk_size,
)


user_recent_items_map = {}
user_last_item_map = {}
item_sim = {}
swing_sim = {}
emb_i2i_sim = {}
article_vec_map = {}
decay_weights = np.power(
    0.7,
    np.arange(max_history_for_sim_sum, dtype=np.float32),
).astype(np.float32)


def log_elapsed(step_name, start_time, df=None):
    message = f'{step_name} done in {time.time() - start_time:.2f}s'
    if df is not None:
        message += f', shape={df.shape}'
    log.info(message)


def optimize_low_cardinality_objects(df, max_unique=64):
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= max_unique:
            df[col] = df[col].astype('category')
    return df


def ensure_sortable_feature_dtypes(df):
    if 'user_id' in df.columns:
        df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(-1).astype('int32')
    if 'article_id' in df.columns:
        df['article_id'] = pd.to_numeric(df['article_id'], errors='coerce').fillna(-1).astype('int32')
    if 'sim_score' in df.columns:
        df['sim_score'] = pd.to_numeric(df['sim_score'], errors='coerce').fillna(0).astype('float32')
    return df


def build_pair_feature(user_ids, article_ids, func, feature_name):
    total = len(user_ids)
    values = np.empty(total, dtype=np.float32)
    start_time = time.time()

    for start in range(0, total, pair_chunk_size):
        end = min(start + pair_chunk_size, total)
        values[start:end] = np.fromiter(
            (func(user_id, article_id) for user_id, article_id in zip(user_ids[start:end], article_ids[start:end])),
            dtype=np.float32,
            count=end - start,
        )
        log.info(
            '%s progress: %s/%s (%.1f%%), elapsed=%.2fs',
            feature_name,
            end,
            total,
            end * 100.0 / total,
            time.time() - start_time,
        )

    return values


def itemcf_sum_for_pair(user_id, article_id):
    interacted_items = user_recent_items_map.get(user_id)
    if not interacted_items:
        return 0.0

    sim_sum = 0.0
    for weight, hist_article_id in zip(decay_weights, interacted_items):
        sim_dict = item_sim.get(hist_article_id)
        if sim_dict is not None:
            sim_sum += sim_dict.get(article_id, 0.0) * float(weight)
    return sim_sum


def itemcf_last_for_pair(user_id, article_id):
    last_item = user_last_item_map.get(user_id)
    if last_item is None:
        return 0.0

    sim_dict = item_sim.get(last_item)
    if sim_dict is None:
        return 0.0
    return float(sim_dict.get(article_id, 0.0))


def swing_last_for_pair(user_id, article_id):
    last_item = user_last_item_map.get(user_id)
    if last_item is None:
        return 0.0

    sim_dict = swing_sim.get(last_item)
    if sim_dict is None:
        return 0.0
    return float(sim_dict.get(article_id, 0.0))


def emb_i2i_last_for_pair(user_id, article_id):
    last_item = user_last_item_map.get(user_id)
    if last_item is None:
        return 0.0

    sim_dict = emb_i2i_sim.get(last_item)
    if sim_dict is None:
        return 0.0
    return float(sim_dict.get(article_id, 0.0))


def cosine_distance(vector1, vector2):
    try:
        if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
            return -1.0
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return -1.0
        return float(np.dot(vector1, vector2) / (norm1 * norm2))
    except Exception:
        return -1.0


def w2w_sum_for_pair(user_id, article_id, num):
    interacted_items = user_recent_items_map.get(user_id)
    if not interacted_items:
        return 0.0

    base_vec = article_vec_map.get(article_id)
    if base_vec is None:
        return 0.0

    sim_sum = 0.0
    for hist_article_id in interacted_items[:num]:
        hist_vec = article_vec_map.get(hist_article_id)
        if hist_vec is not None:
            sim = cosine_distance(base_vec, hist_vec)
            if sim > -1.0:
                sim_sum += sim
    return sim_sum


def w2w_last_for_pair(user_id, article_id):
    last_item = user_last_item_map.get(user_id)
    if last_item is None:
        return 0.0

    base_vec = article_vec_map.get(article_id)
    last_vec = article_vec_map.get(last_item)
    if base_vec is None or last_vec is None:
        return 0.0
    return cosine_distance(base_vec, last_vec)


if __name__ == '__main__':
    if mode == 'valid':
        recall_file = './user_data2/data/offline/recall7.pkl'
        click_file = './user_data2/data/offline/click.pkl'
        sim_base = './user_data2/sim/offline'
        w2v_file = './user_data2/data/offline/article_w2v.pkl'
        save_file = './user_data2/data/offline/feature3.pkl'
    else:
        recall_file = './user_data2/data/online/recall7.pkl'
        click_file = './user_data2/data/online/click.pkl'
        sim_base = './user_data2/sim/online'
        w2v_file = './user_data2/data/online/article_w2v.pkl'
        save_file = './user_data2/data/online/feature3.pkl'

    if not os.path.exists(recall_file):
        raise FileNotFoundError(f'recall7 file not found: {recall_file}')

    step_start = time.time()
    df_feature = pd.read_pickle(recall_file)
    df_click = pd.read_pickle(click_file)
    log_elapsed('load recall7 and click', step_start)

    required_cols = ['user_id', 'article_id', 'sim_score']
    missing_cols = [c for c in required_cols if c not in df_feature.columns]
    if missing_cols:
        raise ValueError(f'recall7 file missing required cols: {missing_cols}, file={recall_file}')

    if 'label' not in df_feature.columns:
        df_feature['label'] = np.nan

    print('optimize df_feature memory...')
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    df_feature = optimize_low_cardinality_objects(df_feature)
    df_feature = ensure_sortable_feature_dtypes(df_feature)

    print('optimize df_click memory...')
    df_click = reduce_mem_usage(df_click, verbose=True)
    df_click = optimize_low_cardinality_objects(df_click)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_click.shape: {df_click.shape}')

    if max_user_candidates > 0:
        step_start = time.time()
        df_feature = (
            df_feature.sort_values(['user_id', 'sim_score'], ascending=[True, False])
            .groupby('user_id', sort=False, group_keys=False)
            .head(max_user_candidates)
            .reset_index(drop=True)
        )
        log_elapsed(f'limit candidates per user to top {max_user_candidates}', step_start, df_feature)

    # 文章特征
    step_start = time.time()
    df_article = pd.read_csv('./data/articles.csv', usecols=['article_id', 'category_id', 'created_at_ts', 'words_count'])
    df_article['created_at_ts'] = (df_article['created_at_ts'] / 1000).astype('int32')
    df_article = reduce_mem_usage(df_article, verbose=False)
    article_meta = df_article.set_index('article_id')

    for col in ['category_id', 'created_at_ts', 'words_count']:
        df_feature[col] = df_feature['article_id'].map(article_meta[col])
    df_feature['created_at_datetime'] = pd.to_datetime(df_feature['created_at_ts'], unit='s', errors='coerce')
    log.debug(f'df_feature.shape after attach article meta: {df_feature.shape}')

    df_click = df_click.sort_values(['user_id', 'click_timestamp']).rename(columns={'click_article_id': 'article_id'}).copy()
    for col in ['category_id', 'created_at_ts', 'words_count']:
        df_click[col] = df_click['article_id'].map(article_meta[col])

    del df_article, article_meta
    gc.collect()

    df_click['click_timestamp'] = (df_click['click_timestamp'] / 1000).astype('int64')
    df_click['click_datetime_hour'] = ((df_click['click_timestamp'] // 3600) % 24).astype('int8')
    log_elapsed('attach article meta and preprocess click table', step_start, df_click)

    # 用户特征
    step_start = time.time()
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby('user_id', sort=False)['created_at_ts'].diff()
    df_click['user_id_click_diff'] = df_click.groupby('user_id', sort=False)['click_timestamp'].diff()
    df_click['click_timestamp_created_at_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']

    user_stats = df_click.groupby('user_id', sort=False).agg(
        user_id_click_article_created_at_ts_diff_mean=('user_id_click_article_created_at_ts_diff', 'mean'),
        user_id_click_diff_mean=('user_id_click_diff', 'mean'),
        user_click_datetime_hour_mean=('click_datetime_hour', 'mean'),
        user_click_datetime_hour_std=('click_datetime_hour', 'std'),
        user_click_timestamp_created_at_ts_diff_mean=('click_timestamp_created_at_ts_diff', 'mean'),
        user_click_timestamp_created_at_ts_diff_std=('click_timestamp_created_at_ts_diff', 'std'),
        user_clicked_article_created_time_max=('created_at_ts', 'max'),
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', 'last'),
        user_click_last_article_created_time=('created_at_ts', 'last'),
        user_click_last_article_click_time=('click_timestamp', 'last'),
        user_clicked_category_nunique=('category_id', 'nunique'),
    )
    user_stats = reduce_mem_usage(user_stats.reset_index(), verbose=False).set_index('user_id')

    for col in user_stats.columns:
        df_feature[col] = df_feature['user_id'].map(user_stats[col])
    log_elapsed('build and map user stats', step_start, df_feature)

    # 候选文章与用户最近行为差异
    step_start = time.time()
    df_feature['user_last_click_created_at_ts_diff'] = (
        df_feature['created_at_ts'] - df_feature['user_click_last_article_created_time']
    )
    df_feature['user_last_click_timestamp_diff'] = (
        df_feature['created_at_ts'] - df_feature['user_click_last_article_click_time']
    )
    df_feature['user_last_click_words_count_diff'] = (
        df_feature['words_count'] - df_feature['user_click_last_article_words_count']
    )
    log_elapsed('build difference features', step_start, df_feature)

    # 点击次数/类目次数
    step_start = time.time()
    user_cnt = df_click['user_id'].value_counts(sort=False)
    article_cnt = df_click['article_id'].value_counts(sort=False)
    category_cnt = df_click['category_id'].value_counts(sort=False)

    df_feature['user_id_cnt'] = df_feature['user_id'].map(user_cnt)
    df_feature['article_id_cnt'] = df_feature['article_id'].map(article_cnt)
    df_feature['category_click_cnt'] = df_feature['category_id'].map(category_cnt).fillna(0)

    user_category_cnt = df_click.groupby(['user_id', 'category_id'], sort=False).size().rename(
        'user_id_category_id_cnt'
    ).reset_index()
    df_feature = df_feature.merge(user_category_cnt, on=['user_id', 'category_id'], how='left')

    user_last_category = df_click.groupby('user_id', sort=False)['category_id'].last()
    df_feature['same_as_last_category'] = (
        df_feature['category_id'] == df_feature['user_id'].map(user_last_category)
    ).astype('int8')

    del user_cnt, article_cnt, category_cnt, user_category_cnt, user_last_category
    gc.collect()
    log_elapsed('build count stats', step_start, df_feature)

    # 用户历史缓存
    step_start = time.time()
    history_window = max(max_history_for_sim_sum, 2)
    user_history = df_click.groupby('user_id', sort=False)['article_id'].agg(list)
    user_recent_items_map = {
        user_id: items[-history_window:][::-1]
        for user_id, items in user_history.items()
    }
    user_last_item_map = {
        user_id: items[0]
        for user_id, items in user_recent_items_map.items()
        if items
    }
    del user_history, user_stats
    gc.collect()
    log_elapsed('build user history cache', step_start)

    user_ids = df_feature['user_id'].to_numpy()
    article_ids = df_feature['article_id'].to_numpy()

    # ItemCF 相似度
    step_start = time.time()
    with open(f'{sim_base}/itemcf_sim.pkl', 'rb') as f:
        item_sim = pickle.load(f)
    log_elapsed('load itemcf sim', step_start)

    df_feature['user_clicked_article_itemcf_sim_sum'] = build_pair_feature(
        user_ids, article_ids, itemcf_sum_for_pair, 'user_clicked_article_itemcf_sim_sum'
    )
    df_feature['user_last_click_article_itemcf_sim'] = build_pair_feature(
        user_ids, article_ids, itemcf_last_for_pair, 'user_last_click_article_itemcf_sim'
    )
    del item_sim
    gc.collect()
    log.info('release itemcf sim cache')

    # Swing 相似度
    swing_path = f'{sim_base}/swing_sim.pkl'
    if os.path.exists(swing_path):
        step_start = time.time()
        with open(swing_path, 'rb') as f:
            swing_sim = pickle.load(f)
        log_elapsed('load swing sim', step_start)

        df_feature['user_last_click_article_swing_sim'] = build_pair_feature(
            user_ids, article_ids, swing_last_for_pair, 'user_last_click_article_swing_sim'
        )
        del swing_sim
        gc.collect()
        log.info('release swing sim cache')
    else:
        log.warning('swing_sim not found: %s, fill with 0', swing_path)
        df_feature['user_last_click_article_swing_sim'] = 0.0

    # embedding 相似度
    emb_i2i_path = f'{sim_base}/emb_i2i_sim.pkl'
    if os.path.exists(emb_i2i_path):
        step_start = time.time()
        with open(emb_i2i_path, 'rb') as f:
            emb_i2i_sim = pickle.load(f)
        log_elapsed('load emb_i2i sim', step_start)

        df_feature['user_last_click_article_emb_i2i_sim'] = build_pair_feature(
            user_ids, article_ids, emb_i2i_last_for_pair, 'user_last_click_article_emb_i2i_sim'
        )
        del emb_i2i_sim
        gc.collect()
        log.info('release emb_i2i sim cache')
    else:
        log.warning('emb_i2i_sim not found: %s, fill with 0', emb_i2i_path)
        df_feature['user_last_click_article_emb_i2i_sim'] = 0.0

    # Word2Vec 相似度
    step_start = time.time()
    with open(w2v_file, 'rb') as f:
        article_vec_map = pickle.load(f)
    log_elapsed('load article w2v cache', step_start)

    df_feature['user_last_click_article_w2v_sim'] = build_pair_feature(
        user_ids, article_ids, w2w_last_for_pair, 'user_last_click_article_w2v_sim'
    )
    df_feature['user_click_article_w2w_sim_sum_2'] = build_pair_feature(
        user_ids,
        article_ids,
        lambda user_id, article_id: w2w_sum_for_pair(user_id, article_id, 2),
        'user_click_article_w2w_sim_sum_2',
    )

    del article_vec_map, df_click
    gc.collect()
    log.info('release article w2v cache')

    keep_columns = [
        'user_id',
        'article_id',
        'label',
        # 文章特征
        'created_at_ts',
        'created_at_datetime',
        'words_count',
        'category_id',
        'article_id_cnt',
        'sim_score',
        'category_click_cnt',
        # 用户特征
        'user_id_cnt',
        'user_id_click_diff_mean',
        'user_click_datetime_hour_std',
        'user_click_datetime_hour_mean',
        'user_clicked_category_nunique',
        'user_id_click_article_created_at_ts_diff_mean',
        'user_click_timestamp_created_at_ts_diff_mean',
        'user_click_timestamp_created_at_ts_diff_std',
        'user_clicked_article_created_time_max',
        'user_clicked_article_words_count_mean',
        'user_click_last_article_words_count',
        'user_click_last_article_created_time',
        'user_click_last_article_click_time',
        # 交互特征
        'user_last_click_created_at_ts_diff',
        'user_last_click_timestamp_diff',
        'user_last_click_words_count_diff',
        'user_id_category_id_cnt',
        'same_as_last_category',
        'user_clicked_article_itemcf_sim_sum',
        'user_last_click_article_itemcf_sim',
        'user_last_click_article_swing_sim',
        'user_last_click_article_w2v_sim',
        'user_click_article_w2w_sim_sum_2',
        'user_last_click_article_emb_i2i_sim',
    ]
    missing_keep_columns = [col for col in keep_columns if col not in df_feature.columns]
    if missing_keep_columns:
        raise ValueError(f'missing expected feature columns: {missing_keep_columns}')
    df_feature = df_feature[keep_columns].copy()

    category_cols = df_feature.select_dtypes(include='category').columns.tolist()
    for col in category_cols:
        df_feature[col] = df_feature[col].cat.codes.astype('int16')

    print('final feature memory optimization...')
    step_start = time.time()
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    df_feature = ensure_sortable_feature_dtypes(df_feature)
    gc.collect()
    log_elapsed('final reduce_mem_usage', step_start, df_feature)

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_feature.to_pickle(save_file)
    log.info('feature saved: %s, shape=%s', save_file, df_feature.shape)
