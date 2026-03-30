import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from lightgbm import early_stopping, log_evaluation

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)


# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm ranker (LambdaRank)')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test_ranker.log')

args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
os.makedirs('./user_data2/model', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'lightgbm ranker，mode: {mode}')


def build_group(df, group_col='user_id'):
    """
    按 group_col 构建 group 数组（每个query下样本数）
    要求：df 已按 group_col 排序
    """
    group = df.groupby(group_col, sort=False).size().values
    return group


def train_model(df_feature, df_query):
    train_idx = df_feature['label'].notnull()
    df_train = df_feature.loc[train_idx].copy()

    test_idx = df_feature['label'].isnull()
    df_test = df_feature.loc[test_idx].copy()

    del df_feature
    gc.collect()

    ycol = 'label'
    drop_cols = [ycol, 'created_at_datetime', 'click_datetime']
    feature_names = [c for c in df_train.columns if c not in drop_cols]
    feature_names.sort()

    # LambdaRank 模型
    model = lgb.LGBMRanker(
        boosting_type='gbdt',
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5, 10, 20, 40, 50],
        num_leaves=64,
        max_depth=10,
        learning_rate=0.05,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=seed,
        importance_type='gain'
    )

    oof = []
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = 0.0
    df_importance_list = []

    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
        kfold.split(df_train[feature_names], df_train[ycol], df_train['user_id'])
    ):
        trn_df = df_train.iloc[trn_idx].copy()
        val_df = df_train.iloc[val_idx].copy()

        # !!! Ranker必须按query分组排序后再构建group
        trn_df.sort_values(['user_id'], inplace=True)
        val_df.sort_values(['user_id'], inplace=True)

        X_train = trn_df[feature_names]
        Y_train = trn_df[ycol].astype(int)

        X_val = val_df[feature_names]
        Y_val = val_df[ycol].astype(int)

        train_group = build_group(trn_df, 'user_id')
        valid_group = build_group(val_df, 'user_id')

        log.debug(f'\nFold_{fold_id + 1} Training ================================\n')

        lgb_model = model.fit(
            X_train,
            Y_train,
            group=train_group,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],  # ← 改这里：加训练集
            eval_group=[train_group, valid_group],          # ← 改这里：两个 group
            eval_names=['train', 'valid'],                   # ← 加命名（可选但推荐）
            eval_at=[5, 10, 20, 40, 50],
            callbacks=[early_stopping(100), log_evaluation(100)]
        )

        # ranker输出打分，直接用于排序
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = val_df[['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        # 测试集预测（为稳妥，也按user_id排序后预测）
        test_tmp = df_test.copy()
        test_tmp.sort_values(['user_id'], inplace=True)
        pred_test = lgb_model.predict(
            test_tmp[feature_names],
            num_iteration=lgb_model.best_iteration_
        )
        test_tmp['pred_fold'] = pred_test

        # 回填到 prediction
        prediction = prediction.merge(
            test_tmp[['user_id', 'article_id', 'pred_fold']],
            on=['user_id', 'article_id'],
            how='left'
        )
        prediction['pred'] += prediction['pred_fold'].fillna(0) / 5
        prediction.drop(columns=['pred_fold'], inplace=True)

        df_importance = pd.DataFrame({
            'feature_name': feature_names,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(lgb_model, f'./user_data2/model/lgb_ranker{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = (
        df_importance.groupby(['feature_name'])['importance']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    log.debug(f'importance: {df_importance.head(50)}')

    # OOF
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])

    os.makedirs('./user_data2/prediction_result', exist_ok=True)
    df_oof.to_csv('./user_data2/prediction_result/oof_lgbm_ranker_valid.csv', index=False)
    prediction.to_csv('./user_data2/prediction_result/test_lgbm_ranker_valid.csv', index=False)

    # 评估
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    df_sub.to_csv('./user_data2/prediction_result/result_valid_lgbm_ranker.csv', index=False)


def online_predict(df_test):
    ycol = 'label'
    drop_cols = [ycol, 'created_at_datetime', 'click_datetime']
    feature_names = [c for c in df_test.columns if c not in drop_cols]
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = 0.0

    # 排序保证和训练一致的query结构
    test_tmp = df_test.copy()
    test_tmp.sort_values(['user_id'], inplace=True)

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'./user_data2/model/lgb_ranker{fold_id}.pkl')
        pred_test = model.predict(test_tmp[feature_names])
        test_tmp[f'pred_{fold_id}'] = pred_test

    pred_cols = [f'pred_{i}' for i in range(5)]
    test_tmp['pred_mean'] = test_tmp[pred_cols].mean(axis=1)

    prediction = prediction.merge(
        test_tmp[['user_id', 'article_id', 'pred_mean']],
        on=['user_id', 'article_id'],
        how='left'
    )
    prediction['pred'] = prediction['pred_mean'].fillna(0)
    prediction.drop(columns=['pred_mean'], inplace=True)

    os.makedirs('./user_data2/prediction_result', exist_ok=True)
    prediction.to_csv('./user_data2/prediction_result/test_lgbm_ranker_online.csv', index=False)

    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    df_sub.to_csv('./user_data2/prediction_result/result_online_lgbm_ranker.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('./user_data2/data/offline/feature3.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        # label 需为整数等级相关性，通常 0/1 即可
        if df_feature['label'].notnull().any():
            df_feature.loc[df_feature['label'].notnull(), 'label'] = (
                df_feature.loc[df_feature['label'].notnull(), 'label'].astype(int)
            )

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('./user_data2/data/online/feature3.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        online_predict(df_feature)