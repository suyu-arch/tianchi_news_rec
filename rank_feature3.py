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

# =========================
# 参数
# =========================
parser = argparse.ArgumentParser(description='排序特征 v3（适配 recall7，无rank/source强依赖）')
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

# 日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(
    '排序特征v3（recall7版），mode=%s, max_user_candidates=%s, max_history_for_sim_sum=%s, pair_chunk_size=%s',
    mode,
    max_user_candidates,
    max_history_for_sim_sum,
    pair_chunk_size,
)

# 全局缓存，供相似度函数读取
user_recent_items_map = {}
user_last_item_map = {}
item_sim = {}
binetwork_sim = {}
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
    # pandas 某些排序路径不支持 float16 作为排序键，统一转成安全类型
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


def binetwork_last_for_pair(user_id, article_id):
    last_item = user_last_item_map.get(user_id)
    if last_item is None:
        return 0.0

    sim_dict = binetwork_sim.get(last_item)
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


# =========================
# 主流程
# =========================
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

    # 读取
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

    print('优化 df_feature 内存...')
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    df_feature = optimize_low_cardinality_objects(df_feature)
    df_feature = ensure_sortable_feature_dtypes(df_feature)

    print('优化 df_click 内存...')
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

    # 文章信息
    step_start = time.time()
    df_article = pd.read_csv('./data/articles.csv', usecols=['article_id', 'category_id', 'created_at_ts', 'words_count'])
    df_article['created_at_ts'] = (df_article['created_at_ts'] / 1000).astype('int32')
    df_article = reduce_mem_usage(df_article, verbose=False)
    article_meta = df_article.set_index('article_id')

    for col in ['category_id', 'created_at_ts', 'words_count']:
        df_feature[col] = df_feature['article_id'].map(article_meta[col])
    log.debug(f'df_feature.shape after attach article meta: {df_feature.shape}')

    df_click = df_click.sort_values(['user_id', 'click_timestamp']).rename(
        columns={'click_article_id': 'article_id'}
    ).copy()
    for col in ['category_id', 'created_at_ts', 'words_count']:
        df_click[col] = df_click['article_id'].map(article_meta[col])

    del df_article, article_meta
    gc.collect()

    df_click['click_timestamp'] = (df_click['click_timestamp'] / 1000).astype('int64')
    df_click['click_datetime_hour'] = ((df_click['click_timestamp'] // 3600) % 24).astype('int8')
    df_click['click_day'] = (df_click['click_timestamp'] // 86400).astype('int32')
    log_elapsed('attach article meta and preprocess click table', step_start, df_click)

    # 用户行为统计特征
    step_start = time.time()
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        'user_id',
        sort=False,
    )['created_at_ts'].diff()
    df_click['user_id_click_diff'] = df_click.groupby(
        'user_id',
        sort=False,
    )['click_timestamp'].diff()
    df_click['click_timestamp_created_at_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']

    user_stats = df_click.groupby('user_id', sort=False).agg(
        user_id_click_article_created_at_ts_diff_mean=('user_id_click_article_created_at_ts_diff', 'mean'),
        user_id_click_diff_mean=('user_id_click_diff', 'mean'),
        user_click_timestamp_created_at_ts_diff_mean=('click_timestamp_created_at_ts_diff', 'mean'),
        user_click_timestamp_created_at_ts_diff_std=('click_timestamp_created_at_ts_diff', 'std'),
        user_click_datetime_hour_std=('click_datetime_hour', 'std'),
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', 'last'),
        user_click_last_article_created_time=('created_at_ts', 'last'),
        user_clicked_article_created_time_max=('created_at_ts', 'max'),
        user_click_last_article_click_time=('click_timestamp', 'last'),
        user_clicked_article_click_time_mean=('click_timestamp', 'mean'),
    )

    user_stats_extra = df_click.groupby('user_id', sort=False).agg(
        user_click_first_ts=('click_timestamp', 'min'),
        user_click_last_ts=('click_timestamp', 'max'),
        user_click_active_days=('click_day', 'nunique'),
        user_clicked_category_nunique=('category_id', 'nunique'),
    )
    user_stats_extra['user_click_span_hours'] = (
        (user_stats_extra['user_click_last_ts'] - user_stats_extra['user_click_first_ts']) / 3600.0
    ).astype(np.float32)

    user_stats = user_stats.join(
        user_stats_extra[['user_click_active_days', 'user_clicked_category_nunique', 'user_click_span_hours']],
        how='left'
    )

    user_stats = reduce_mem_usage(user_stats.reset_index(), verbose=False).set_index('user_id')

    for col in user_stats.columns:
        df_feature[col] = df_feature['user_id'].map(user_stats[col])
    log_elapsed('build and map user stats', step_start, df_feature)

    # 候选与用户行为差值
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

    now_ts = float(df_click['click_timestamp'].max())
    df_feature['candidate_age_hours'] = ((now_ts - df_feature['created_at_ts']).clip(lower=0) / 3600.0).astype(np.float32)
    df_feature['user_last_click_recency_hours'] = (
        (now_ts - df_feature['user_click_last_article_click_time']).clip(lower=0) / 3600.0
    ).astype(np.float32)

    df_feature['is_fresh_24h'] = (df_feature['candidate_age_hours'] <= 24).astype('int8')
    df_feature['is_fresh_72h'] = (df_feature['candidate_age_hours'] <= 72).astype('int8')
    df_feature['is_fresh_7d'] = (df_feature['candidate_age_hours'] <= 24 * 7).astype('int8')
    log_elapsed('build difference and freshness features', step_start, df_feature)

    # 计数统计
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
    df_feature['user_last_category_id'] = df_feature['user_id'].map(user_last_category)
    df_feature['same_as_last_category'] = (
        df_feature['category_id'] == df_feature['user_last_category_id']
    ).astype('int8')
    df_feature.drop(columns=['user_last_category_id'], inplace=True)

    del user_cnt, article_cnt, category_cnt, user_category_cnt, user_last_category
    gc.collect()
    log_elapsed('build count stats', step_start, df_feature)

    # sim_score 统计特征
    step_start = time.time()
    g = df_feature.groupby('user_id', sort=False)['sim_score']
    df_feature['user_cand_cnt'] = g.transform('size')
    df_feature['sim_score_user_mean'] = g.transform('mean')
    df_feature['sim_score_user_std'] = g.transform('std').fillna(0)
    df_feature['sim_score_user_min'] = g.transform('min')
    df_feature['sim_score_user_max'] = g.transform('max')
    df_feature['sim_score_user_median'] = g.transform('median')
    df_feature['sim_score_user_q25'] = g.transform(lambda x: x.quantile(0.25))
    df_feature['sim_score_user_q75'] = g.transform(lambda x: x.quantile(0.75))
    df_feature['sim_score_user_iqr'] = df_feature['sim_score_user_q75'] - df_feature['sim_score_user_q25']

    df_feature['sim_score_z'] = (
        (df_feature['sim_score'] - df_feature['sim_score_user_mean']) /
        (df_feature['sim_score_user_std'] + 1e-6)
    )
    df_feature['sim_score_minmax'] = (
        (df_feature['sim_score'] - df_feature['sim_score_user_min']) /
        (df_feature['sim_score_user_max'] - df_feature['sim_score_user_min'] + 1e-6)
    )
    df_feature['sim_score_to_user_max_ratio'] = (
        df_feature['sim_score'] / (df_feature['sim_score_user_max'] + 1e-6)
    )
    df_feature['sim_score_to_user_median_diff'] = (
        df_feature['sim_score'] - df_feature['sim_score_user_median']
    )
    df_feature['user_cand_score_sum'] = g.transform('sum')
    df_feature['user_cand_score_cv'] = (
        df_feature['sim_score_user_std'] / (df_feature['sim_score_user_mean'] + 1e-6)
    )
    log_elapsed('build sim-score stats', step_start, df_feature)

    # 排名竞争度（不使用 rank 列）
    step_start = time.time()
    df_feature = ensure_sortable_feature_dtypes(df_feature)
    df_feature = df_feature.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    g_sorted = df_feature.groupby('user_id', sort=False)['sim_score']
    df_feature['sim_score_prev'] = g_sorted.shift(1)
    df_feature['sim_score_next'] = g_sorted.shift(-1)
    df_feature['sim_score_top1'] = g_sorted.transform('max')

    df_feature['sim_score_gap_prev'] = (df_feature['sim_score_prev'] - df_feature['sim_score']).fillna(0)
    df_feature['sim_score_gap_next'] = (df_feature['sim_score'] - df_feature['sim_score_next']).fillna(0)
    df_feature['sim_score_gap_top1'] = (df_feature['sim_score_top1'] - df_feature['sim_score']).fillna(0)

    df_feature.drop(columns=['sim_score_prev', 'sim_score_next', 'sim_score_top1'], inplace=True)
    log_elapsed('build score gap features', step_start, df_feature)

    # 交叉特征
    df_feature['user_act_x_item_pop'] = df_feature['user_id_cnt'] * np.log1p(df_feature['article_id_cnt'])
    df_feature['score_x_item_pop'] = df_feature['sim_score'] * np.log1p(df_feature['article_id_cnt'])
    df_feature['score_x_user_act'] = df_feature['sim_score'] * np.log1p(df_feature['user_id_cnt'])
    df_feature['score_z_x_item_pop'] = df_feature['sim_score_z'] * np.log1p(df_feature['article_id_cnt'])

    df_feature['fresh_x_user_recency'] = (
        df_feature['is_fresh_24h'] * (1.0 / (1.0 + df_feature['user_last_click_recency_hours']))
    )
    df_feature['fresh_x_category_hot'] = (
        (df_feature['candidate_age_hours'].clip(upper=168) / 168.0) * np.log1p(df_feature['category_click_cnt'])
    )
    df_feature['fresh_rarity_in_user_cand'] = (
        df_feature['is_fresh_24h'] /
        (df_feature.groupby('user_id', sort=False)['is_fresh_24h'].transform('sum') + 1.0)
    )
    df_feature['user_span_x_article_age'] = (
        np.log1p(df_feature['user_click_span_hours'].clip(lower=0)) *
        np.log1p(df_feature['candidate_age_hours'].clip(lower=0))
    )
    df_feature['user_active_days_x_article_pop'] = (
        df_feature['user_click_active_days'] * np.log1p(df_feature['article_id_cnt'])
    )
    df_feature['user_cat_diversity_x_cat_match'] = (
        df_feature['user_clicked_category_nunique'] * df_feature['same_as_last_category']
    )

    # 构建用户历史缓存
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

    # itemcf
    step_start = time.time()
    with open(f'{sim_base}/itemcf_sim.pkl', 'rb') as f:
        item_sim = pickle.load(f)
    log_elapsed('load itemcf sim', step_start)

    df_feature['user_clicked_article_itemcf_sim_sum'] = build_pair_feature(
        user_ids,
        article_ids,
        itemcf_sum_for_pair,
        'user_clicked_article_itemcf_sim_sum',
    )
    df_feature['user_last_click_article_itemcf_sim'] = build_pair_feature(
        user_ids,
        article_ids,
        itemcf_last_for_pair,
        'user_last_click_article_itemcf_sim',
    )
    del item_sim
    gc.collect()
    log.info('release itemcf sim cache')

    # binetwork
    step_start = time.time()
    with open(f'{sim_base}/binetwork_sim.pkl', 'rb') as f:
        binetwork_sim = pickle.load(f)
    log_elapsed('load binetwork sim', step_start)

    df_feature['user_last_click_article_binetwork_sim'] = build_pair_feature(
        user_ids,
        article_ids,
        binetwork_last_for_pair,
        'user_last_click_article_binetwork_sim',
    )
    del binetwork_sim
    gc.collect()
    log.info('release binetwork sim cache')

    # swing
    swing_path = f'{sim_base}/swing_sim.pkl'
    if os.path.exists(swing_path):
        step_start = time.time()
        with open(swing_path, 'rb') as f:
            swing_sim = pickle.load(f)
        log_elapsed('load swing sim', step_start)

        df_feature['user_last_click_article_swing_sim'] = build_pair_feature(
            user_ids,
            article_ids,
            swing_last_for_pair,
            'user_last_click_article_swing_sim',
        )
        del swing_sim
        gc.collect()
        log.info('release swing sim cache')
    else:
        log.warning('swing_sim 不存在: %s，用0填充', swing_path)
        df_feature['user_last_click_article_swing_sim'] = 0.0

    # emb_i2i
    emb_i2i_path = f'{sim_base}/emb_i2i_sim.pkl'
    if os.path.exists(emb_i2i_path):
        step_start = time.time()
        with open(emb_i2i_path, 'rb') as f:
            emb_i2i_sim = pickle.load(f)
        log_elapsed('load emb_i2i sim', step_start)

        df_feature['user_last_click_article_emb_i2i_sim'] = build_pair_feature(
            user_ids,
            article_ids,
            emb_i2i_last_for_pair,
            'user_last_click_article_emb_i2i_sim',
        )
        del emb_i2i_sim
        gc.collect()
        log.info('release emb_i2i sim cache')
    else:
        log.warning('emb_i2i_sim 不存在: %s，用0填充', emb_i2i_path)
        df_feature['user_last_click_article_emb_i2i_sim'] = 0.0

    # w2v
    step_start = time.time()
    with open(w2v_file, 'rb') as f:
        article_vec_map = pickle.load(f)
    log_elapsed('load article w2v cache', step_start)

    df_feature['user_last_click_article_w2v_sim'] = build_pair_feature(
        user_ids,
        article_ids,
        w2w_last_for_pair,
        'user_last_click_article_w2v_sim',
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

    # 保存前把 category 转成整数编码
    category_cols = df_feature.select_dtypes(include='category').columns.tolist()
    for col in category_cols:
        df_feature[col] = df_feature[col].cat.codes.astype('int16')

    # 最终内存优化
    print('开始最终内存优化...')
    step_start = time.time()
    df_feature = reduce_mem_usage(df_feature, verbose=True)
    df_feature = ensure_sortable_feature_dtypes(df_feature)
    gc.collect()
    log_elapsed('final reduce_mem_usage', step_start, df_feature)

    # 保存
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_feature.to_pickle(save_file)
    log.info('特征已保存: %s, shape=%s', save_file, df_feature.shape)
