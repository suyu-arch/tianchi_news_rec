import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from utils import Logger, evaluate, gen_sub

seed = 2020
np.random.seed(seed)

parser = argparse.ArgumentParser(description='ranking fusion / stacking')
parser.add_argument('--mode', default='valid', choices=['valid', 'online'])
parser.add_argument('--logfile', default='stacking.log')
parser.add_argument(
    '--weights',
    default='',
    help='comma separated weights for lgbm,lgbm_ranker,din, e.g. 0.5,0.5,0.0',
)

args, unknown = parser.parse_known_args()
mode = args.mode

os.makedirs('./user_data2/log', exist_ok=True)
os.makedirs('./user_data2/prediction_result', exist_ok=True)
log = Logger(f'./user_data2/log/{args.logfile}').logger


def pick_existing(paths, desc):
    for path in paths:
        if os.path.exists(path):
            log.info(f'use {desc}: {path}')
            return path
    raise FileNotFoundError(f'no available file for {desc}: {paths}')


def parse_weights(text):
    if not text:
        return None

    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != 3:
        raise ValueError('--weights must contain exactly 3 comma separated numbers')

    weights = tuple(float(p) for p in parts)
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f'weights must sum to 1, got {weights}')
    return weights


def norm_by_user(df, pred_col):
    out = df.copy()
    out[pred_col] = out.groupby('user_id')[pred_col].rank(method='average', pct=True)
    return out


def load_and_merge_oof():
    oof_lgb = pd.read_csv('./user_data2/prediction_result/oof_lgbm_valid.csv')
    oof_rk = pd.read_csv('./user_data2/prediction_result/oof_lgbm_ranker_valid.csv')
    din_oof_path = pick_existing(
        [
            './user_data2/prediction_result/oof_din_valid3.csv',
            './user_data2/prediction_result/oof_din_valid2.csv',
            './user_data2/prediction_result/oof_din_valid.csv',
        ],
        'DIN OOF',
    )
    oof_din = pd.read_csv(din_oof_path)

    oof_lgb = oof_lgb.rename(columns={'pred': 'pred_lgb'})
    oof_rk = oof_rk.rename(columns={'pred': 'pred_ranker'})
    oof_din = oof_din.rename(columns={'pred': 'pred_din'})

    keys = ['user_id', 'article_id', 'label']
    df = oof_lgb[keys + ['pred_lgb']].merge(
        oof_rk[keys + ['pred_ranker']],
        on=keys,
        how='inner',
    ).merge(
        oof_din[keys + ['pred_din']],
        on=keys,
        how='inner',
    )

    for col in ['pred_lgb', 'pred_ranker', 'pred_din']:
        df = norm_by_user(df, col)

    return df


def load_and_merge_test(which='valid'):
    if which == 'valid':
        lgb_path = './user_data2/prediction_result/test_lgbm_valid.csv'
        ranker_path = './user_data2/prediction_result/test_lgbm_ranker_valid.csv'
        din_path = pick_existing(
            [
                './user_data2/prediction_result/test_din_valid3.csv',
                './user_data2/prediction_result/test_din_valid2.csv',
                './user_data2/prediction_result/test_din_valid.csv',
            ],
            'DIN valid prediction',
        )
    else:
        lgb_path = './user_data2/prediction_result/test_lgbm_online.csv'
        ranker_path = './user_data2/prediction_result/test_lgbm_ranker_online.csv'
        din_path = pick_existing(
            [
                './user_data2/prediction_result/test_din_online3.csv',
                './user_data2/prediction_result/test_din_online2.csv',
                './user_data2/prediction_result/test_din_online.csv',
            ],
            'DIN online prediction',
        )

    t_lgb = pd.read_csv(lgb_path).rename(columns={'pred': 'pred_lgb'})
    t_rk = pd.read_csv(ranker_path).rename(columns={'pred': 'pred_ranker'})
    t_din = pd.read_csv(din_path).rename(columns={'pred': 'pred_din'})

    keys = ['user_id', 'article_id']
    df = t_lgb[keys + ['pred_lgb']].merge(
        t_rk[keys + ['pred_ranker']],
        on=keys,
        how='inner',
    ).merge(
        t_din[keys + ['pred_din']],
        on=keys,
        how='inner',
    )

    for col in ['pred_lgb', 'pred_ranker', 'pred_din']:
        df = norm_by_user(df, col)

    return df


def evaluate_weighted(oof_df, weights, df_query):
    w_lgb, w_rk, w_din = weights

    oof = oof_df[['user_id', 'article_id', 'label']].copy()
    oof['pred'] = (
        w_lgb * oof_df['pred_lgb'] +
        w_rk * oof_df['pred_ranker'] +
        w_din * oof_df['pred_din']
    )

    oof = oof.sort_values(['user_id', 'pred'], ascending=[True, False])
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(oof, total)

    score = 0.8 * mrr_5 + 0.2 * hitrate_5
    return {
        'weights': weights,
        'score': score,
        'mrr_5': mrr_5,
        'hitrate_5': hitrate_5,
        'hitrate_10': hitrate_10,
        'hitrate_20': hitrate_20,
    }


def grid_search_weights(oof_df, df_query, step=0.05):
    log.info('=' * 60)
    log.info('Starting Grid Search for Best Weights...')
    log.info(f'Search step: {step}')

    best_result = None
    best_score = -1
    results = []
    candidates = []

    for w_lgb in np.arange(0, 1 + step, step):
        for w_rk in np.arange(0, 1 + step, step):
            w_din = 1 - w_lgb - w_rk
            if 0 <= w_din <= 1:
                candidates.append((round(float(w_lgb), 4), round(float(w_rk), 4), round(float(w_din), 4)))

    log.info(f'Total candidates: {len(candidates)}')

    for i, weights in enumerate(candidates, 1):
        result = evaluate_weighted(oof_df, weights, df_query)
        results.append(result)

        if result['score'] > best_score:
            best_score = result['score']
            best_result = result

        if i % 50 == 0:
            log.info(f'  Progress: {i}/{len(candidates)}, current best: {best_score:.4f}')

    results.sort(key=lambda x: x['score'], reverse=True)
    log.info('-' * 60)
    log.info('Top 10 Weight Combinations:')
    for idx, row in enumerate(results[:10], 1):
        log.info(
            f"  {idx}. weights={row['weights']}, score={row['score']:.4f}, "
            f"MRR@5={row['mrr_5']:.4f}, HR@5={row['hitrate_5']:.4f}"
        )

    log.info('-' * 60)
    log.info(f"BEST WEIGHTS: {best_result['weights']}")
    log.info(f"  Score: {best_result['score']:.4f}")
    log.info(f"  MRR@5: {best_result['mrr_5']:.4f}")
    log.info(f"  HitRate@5: {best_result['hitrate_5']:.4f}")
    log.info('=' * 60)

    return best_result['weights'], results


def weighted_fusion(oof_df, test_df, df_query, weights=None):
    if weights is None:
        weights = (0.35, 0.45, 0.20)
        log.info(f'Using empirical weights: {weights}')
    else:
        log.info(f'Using optimized weights: {weights}')

    w_lgb, w_rk, w_din = weights

    oof = oof_df[['user_id', 'article_id', 'label']].copy()
    oof['pred'] = (
        w_lgb * oof_df['pred_lgb'] +
        w_rk * oof_df['pred_ranker'] +
        w_din * oof_df['pred_din']
    )

    test_pred = test_df[['user_id', 'article_id']].copy()
    test_pred['pred'] = (
        w_lgb * test_df['pred_lgb'] +
        w_rk * test_df['pred_ranker'] +
        w_din * test_df['pred_din']
    )

    oof = oof.sort_values(['user_id', 'pred'], ascending=[True, False])
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(oof, total)

    log.info('[Weighted Fusion] Final Metrics:')
    log.info(f'  HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}')
    log.info(f'  HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}')
    log.info(f'  HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}')

    test_pred.to_csv('./user_data2/prediction_result/test_stacking_weighted_valid.csv', index=False)
    sub = gen_sub(test_pred).sort_values(['user_id'])
    sub.to_csv('./user_data2/prediction_result/result_valid_stacking_weighted.csv', index=False)
    return weights


def weighted_fusion_online(test_df, weights=None):
    if weights is None:
        weights = (0.5, 0.5, 0.0)
        log.info(f'Using default online weights: {weights}')
    else:
        log.info(f'Using provided online weights: {weights}')

    w_lgb, w_rk, w_din = weights
    test_pred = test_df[['user_id', 'article_id']].copy()
    test_pred['pred'] = (
        w_lgb * test_df['pred_lgb'] +
        w_rk * test_df['pred_ranker'] +
        w_din * test_df['pred_din']
    )

    test_pred.to_csv('./user_data2/prediction_result/test_stacking_weighted_online.csv', index=False)
    sub = gen_sub(test_pred).sort_values(['user_id'])
    sub.to_csv('./user_data2/prediction_result/result_online_stacking_weighted.csv', index=False)
    return weights


def lr_stacking(oof_df, test_df, df_query, n_splits=5):
    feats = ['pred_lgb', 'pred_ranker', 'pred_din']

    X = oof_df[feats].values
    y = oof_df['label'].astype(int).values
    groups = oof_df['user_id'].values

    oof_meta = np.zeros(len(oof_df), dtype=np.float32)
    test_meta = np.zeros(len(test_df), dtype=np.float32)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        x_trn, y_trn = X[trn_idx], y[trn_idx]
        x_val = X[val_idx]

        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200, random_state=seed)
        clf.fit(x_trn, y_trn)

        oof_meta[val_idx] = clf.predict_proba(x_val)[:, 1]
        test_meta += clf.predict_proba(test_df[feats].values)[:, 1] / n_splits

        log.info(f'[LR-Stack] fold={fold} coef={clf.coef_.tolist()}')

    oof = oof_df[['user_id', 'article_id', 'label']].copy()
    oof['pred'] = oof_meta
    oof = oof.sort_values(['user_id', 'pred'], ascending=[True, False])

    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(oof, total)

    log.info('[LR Stacking] Final Metrics:')
    log.info(f'  HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}')
    log.info(f'  HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}')
    log.info(f'  HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}')

    test_pred = test_df[['user_id', 'article_id']].copy()
    test_pred['pred'] = test_meta
    test_pred.to_csv('./user_data2/prediction_result/test_stacking_lr_valid.csv', index=False)

    sub = gen_sub(test_pred).sort_values(['user_id'])
    sub.to_csv('./user_data2/prediction_result/result_valid_stacking_lr.csv', index=False)


if __name__ == '__main__':
    explicit_weights = parse_weights(args.weights)

    if mode == 'valid':
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')
        oof_df = load_and_merge_oof()
        test_df = load_and_merge_test('valid')
        log.info(f'oof size={oof_df.shape}, test size={test_df.shape}')

        if explicit_weights is None:
            log.info('Running grid search for optimal weights...')
            best_weights, all_results = grid_search_weights(oof_df, df_query, step=0.05)
        else:
            best_weights = explicit_weights
            log.info(f'Skip grid search, use explicit weights: {best_weights}')

        log.info('Running weighted fusion with optimized weights...')
        weighted_fusion(oof_df, test_df, df_query, weights=best_weights)

        if explicit_weights is None:
            log.info('Running weighted fusion with empirical weights (for comparison)...')
            weighted_fusion(oof_df, test_df, df_query, weights=None)

            log.info('Running LR stacking...')
            lr_stacking(oof_df, test_df, df_query, n_splits=5)

        log.info('All stacking methods completed!')
    else:
        test_df = load_and_merge_test('online')
        log.info(f'online test size={test_df.shape}')
        weighted_fusion_online(test_df, weights=explicit_weights)
        log.info('Online weighted fusion completed!')
