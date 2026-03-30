import argparse
import os
import random
from pathlib import Path
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

random.seed(2020)

parser = argparse.ArgumentParser(description='data preparation')
parser.add_argument('--mode', default='valid', choices=['valid', 'online'])
parser.add_argument('--logfile', default='test.log')

args, unknown = parser.parse_known_args()
mode = args.mode
logfile = args.logfile

os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'data prepare, mode={mode}')


def data_offline(df_train_click, df_test_click):
    train_users = df_train_click['user_id'].unique().tolist()
    val_user_count = min(50000, len(train_users))
    val_users = sample(train_users, val_user_count)
    val_users_set = set(val_users)

    log.debug(f'val_users num: {len(val_users_set)}')

    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby('user_id')
    for user_id, g in tqdm(groups):
        if user_id in val_users_set:
            valid_query = g.tail(1)
            valid_query_list.append(valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)
            if len(train_click) > 0:
                click_list.append(train_click)
        else:
            click_list.append(g)

    if len(click_list) == 0:
        raise ValueError('training clicks are empty')
    if len(valid_query_list) == 0:
        raise ValueError('validation queries are empty')

    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    df_test_query = pd.DataFrame(
        [[user, -1] for user in tqdm(test_users)],
        columns=['user_id', 'click_article_id'],
    )

    df_query = pd.concat([df_valid_query, df_test_query], sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click], sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)

    log.debug(f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')

    os.makedirs('./user_data2/data/offline', exist_ok=True)
    df_click.to_pickle('./user_data2/data/offline/click.pkl')
    df_query.to_pickle('./user_data2/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    df_test_query = pd.DataFrame(
        [[user, -1] for user in tqdm(test_users)],
        columns=['user_id', 'click_article_id'],
    )

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click], sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)

    log.debug(f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    os.makedirs('./user_data2/data/online', exist_ok=True)
    df_click.to_pickle('./user_data2/data/online/click.pkl')
    df_query.to_pickle('./user_data2/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('./data/train_click_log.csv')
    df_test_click = pd.read_csv('./data/testA_click_log.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
