import argparse
import os
import pickle
import logging
import sys

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==================== 新增：参数解析 ====================
parser = argparse.ArgumentParser(description='build emb_i2i_sim from article embeddings')
parser.add_argument('--emb_path', default='./data/articles_emb.csv', help='csv或pkl(dict)')
parser.add_argument('--save_path', default='./user_data2/sim/offline/emb_i2i_sim.pkl')
parser.add_argument('--topk', type=int, default=200)

# 新增缺失的参数
parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'],
                    help='运行模式，影响日志记录')
parser.add_argument('--logfile', type=str, default=None,
                    help='日志文件路径，默认输出到控制台')

args = parser.parse_args()

# ==================== 新增：日志设置 ====================
def setup_logging(logfile=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        if not os.path.dirname(logfile):
            logfile = os.path.join('./user_data2/log', logfile)
        os.makedirs(os.path.dirname(logfile) if os.path.dirname(logfile) else '.', exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

log = setup_logging(args.logfile)
log.info(f'Starting embedding_sim.py, mode={args.mode}')

# ==================== 修复：加载函数 ====================
def load_article_emb(path):
    log.info(f'Loading embeddings from: {path}')
    
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)  # dict{article_id: vec}
        article_ids, vecs = [], []
        for k, v in obj.items():
            article_ids.append(int(k))
            # 确保每个向量都是numpy数组且类型正确
            vec = np.asarray(v, dtype=np.float32)
            vecs.append(vec)
        
        # 修复：使用vstack后确保contiguous
        vecs = np.vstack(vecs)
        log.info(f'Loaded {len(article_ids)} items from pickle')
        return article_ids, vecs
    
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        assert 'article_id' in df.columns, "CSV must contain 'article_id' column"
        emb_cols = [c for c in df.columns if c != 'article_id']
        
        article_ids = df['article_id'].astype(int).tolist()
        vecs = df[emb_cols].values  # .values 返回的通常是C-contiguous
        
        log.info(f'Loaded {len(article_ids)} items from CSV, dim={len(emb_cols)}')
        return article_ids, vecs
    
    else:
        raise ValueError(f'Unsupported file format: {path}')

# ==================== 修复：主逻辑 ====================
try:
    article_ids, vecs = load_article_emb(args.emb_path)
    
    # 关键修复：双重保险确保数组是C-contiguous且类型正确
    if vecs.dtype != np.float32:
        log.info(f'Converting dtype from {vecs.dtype} to float32')
        vecs = vecs.astype(np.float32)
    
    if not vecs.flags['C_CONTIGUOUS']:
        log.info('Converting to C-contiguous array')
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    else:
        # 即使已经是contiguous，也确保类型并创建新数组避免内存问题
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    
    log.info(f'Vecs shape: {vecs.shape}, dtype: {vecs.dtype}, is_c_contiguous: {vecs.flags["C_CONTIGUOUS"]}')
    
    # 归一化
    faiss.normalize_L2(vecs)
    log.info('L2 normalization completed')
    
    # 构建索引
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    log.info(f'FAISS index built: {index.ntotal} vectors, dim={dim}')
    
    # 搜索相似度
    log.info(f'Searching top-{args.topk+1} similarities...')
    sims, idxs = index.search(vecs, args.topk + 1)  # 包含自己
    
    # 构建相似度字典
    idx2aid = dict(enumerate(article_ids))
    emb_i2i_sim = {}
    
    for i in tqdm(range(len(article_ids)), desc='build sim dict'):
        aid = article_ids[i]
        emb_i2i_sim[aid] = {}
        
        for s, j in zip(sims[i], idxs[i]):
            if j < 0:
                continue
            rid = idx2aid.get(j)
            if rid is None or rid == aid:
                continue
            emb_i2i_sim[aid][rid] = float(s)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(emb_i2i_sim, f)
    
    log.info(f'Saved: {args.save_path}, items={len(emb_i2i_sim)}')
    log.info(f'Mode {args.mode} completed successfully')

except Exception as e:
    log.error(f'Error occurred: {e}', exc_info=True)
    raise
