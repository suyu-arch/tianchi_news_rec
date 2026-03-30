import argparse
import math
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
import faiss
#from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate
warnings.filterwarnings('ignore')


max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args, unknown = parser.parse_known_args() 

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data2/log', exist_ok=True)
log = Logger(f'./user_data2/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


'''训练Word2Vec模型，构建文章向量映射表'''
def word2vec(df_, f1, f2, model_path):#f1序列ID（userID），f2序列内容(itemID)，model_path模型保存路径
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    #按照f1（用户ID）分组，聚合f2（文章ID）为列表，构建用户点击序列，结果是一个DataFrame，每行包含一个用户ID和该用户点击的文章ID列表

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    #提取用户点击的文章ID列表，构建成一个二维列表，每个子列表是一个用户的点击序列，Word2Vec 的训练语料
    del tmp['{}_{}_list'.format(f1, f2)]#删除临时列，释放内存

    words = []
    for i in range(len(sentences)):#遍历每个用户的点击序列，将文章ID转换为字符串（Word2Vec要求输入是字符串），并收集所有文章ID构建词汇表
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')#如果模型文件存在，直接加载模型，避免重复训练
    else:
        model = Word2Vec(sentences=sentences,
                         vector_size=256,#词向量维度设置为256
                         window=3,#上下文窗口大小设置为3，表示在训练过程中，每个目标词会考虑前后3个词作为上下文
                         min_count=1,#最小词频设置为1，表示所有出现过的文章ID都会被纳入训练，不会被忽略
                         sg=1,#使用Skip-gram模型进行训练，适合小语料和稀疏数据，能够更好地捕捉文章之间的相似关系
                         hs=0,#不使用层次Softmax，配合负采样提高训练效率
                         seed=seed,#固定随机种子，保证训练结果可复现
                         negative=5,#负采样数量设置为5，表示在训练过程中，每个正样本会配合5个负样本进行训练，帮助模型更好地区分相似和不相似的文章
                         workers=10,#使用10个线程进行训练，提升训练速度
                         epochs=1)#训练迭代次数设置为1，快速训练得到初始的文章向量表示，后续可以根据需要增加迭代次数进行微调
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]
            #将训练得到的文章向量存储在一个字典中，键是文章ID（转换为整数），值是对应的词向量，方便后续召回阶段使用

    return article_vec_map


@multitasking.task
# max_threads = multitasking.config['CPU_CORES']
# multitasking.set_max_threads(max_threads)
# multitasking.set_engine('process')
def recall(df_query, article_vec_map, article_index,index_to_article, user_item_dict,
           worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]
        # 取用户最近点击的1篇文章，进行相似文章召回，w2v只取最近点击的1篇文章进行召回，进一步突出时序性，减少噪声

        for item in interacted_items:
            article_vec = article_vec_map[item]

            # item_ids, distances = article_index.get_nns_by_vector(
            #     article_vec, 100, include_distances=True)
            # #使用FAISS索引，根据文章向量快速检索出与该文章最相似的100篇文章，返回文章ID列表和对应的距离列表，距离越小表示越相似
            # sim_scores = [2 - distance for distance in distances]
            # #将距离转换为相似度分数，距离越小相似度越高，这里简单地使用2减去距离作为相似度分数，具体的转换方式可以根据实际情况调整
            query_vec = np.array([article_vec]).astype('float32')#转换为FAISS输入格式
            faiss.normalize_L2(query_vec)#向量归一化，使得内积等于余弦相似度

            distances, item_ids = article_index.search(query_vec, 100)#使用FAISS索引，根据文章向量快速检索出与该文章最相似的100篇文章，返回文章ID列表和对应的距离列表，距离代表余弦相似度

            item_ids = item_ids[0]#FAISS返回的item_ids是二维数组，取第一行作为结果，因为我们只查询了一个向量
            distances = distances[0]

            # index -> article_id
            item_ids = [index_to_article[i] for i in item_ids]#将FAISS返回的索引转换回文章ID，index_to_article是一个列表或字典，存储了索引到文章ID的映射关系

            sim_scores = distances#FAISS返回的距离已经是归一化后的内积，
            
            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]# 取最终召回分数最高的50篇文章作为候选集reverse=True
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('./user_data2/tmp/w2v', exist_ok=True)
    df_data.to_pickle('./user_data2/tmp/w2v/{}.pkl'.format(worker_id))



if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/offline/query.pkl')

        os.makedirs('./user_data2/data/offline', exist_ok=True)
        os.makedirs('./user_data2/model/offline', exist_ok=True)

        w2v_file = './user_data2/data/offline/article_w2v.pkl'
        model_path = './user_data2/model/offline'
    else:
        df_click = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data2/data/online/query.pkl')

        os.makedirs('./user_data2/data/online', exist_ok=True)
        os.makedirs('./user_data2/model/online', exist_ok=True)

        w2v_file = './user_data2/data/online/article_w2v.pkl'
        model_path = './user_data2/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    # article_index = AnnoyIndex(256, 'angular')
    # article_index.set_seed(2020)

    # for article_id, emb in tqdm(article_vec_map.items()):
    #     article_index.add_item(article_id, emb)

    # article_index.build(100)
    dim = 256

    article_ids = []
    article_vecs = []

    for article_id, emb in article_vec_map.items():
        article_ids.append(article_id)
        article_vecs.append(emb)

    article_vecs = np.array(article_vecs).astype('float32')

    # 归一化（模拟 Annoy angular）
    faiss.normalize_L2(article_vecs)

    article_index = faiss.IndexFlatIP(dim)#计算内积的索引，适合已经归一化的向量，内积等价于余弦相似度
    article_index.add(article_vecs)#将文章向量添加到FAISS索引中，准备进行相似文章的快速检索

    # index -> article_id 映射
    index_to_article = dict(enumerate(article_ids))
    #建立索引到文章ID的映射关系，FAISS返回的结果是索引，需要通过这个映射关系转换回实际的文章ID，index_to_article是一个字典，键是索引，值是对应的文章ID

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('./user_data2/tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, index_to_article, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data2/tmp/w2v'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp])

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('./user_data2/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('./user_data2/data/online/recall_w2v.pkl')