import argparse
import os
import pickle
import random
import warnings
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import evaluate

warnings.filterwarnings('ignore')

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2020)


# =========================
# 参数
# =========================
parser = argparse.ArgumentParser(description='PyTorch YouTubeDNN 召回（最终版）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--seq_max_len', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--recall_k', type=int, default=50)
parser.add_argument('--gpu_id', type=int, default=0, help='FAISS GPU设备ID')
parser.add_argument('--use_faiss_gpu', action='store_true', help='使用FAISS GPU')

args, unknown = parser.parse_known_args()

mode = args.mode
device = torch.device(args.device)
gpu_id = args.gpu_id


# =========================
# 日志
# =========================
class SimpleLogger:
    def __init__(self, logfile):
        import logging
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.logger = logging.getLogger(f'youtubeDNN_{logfile}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if len(self.logger.handlers) == 0:
            fh = logging.FileHandler(logfile, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)


os.makedirs('./user_data2/log', exist_ok=True)
log = SimpleLogger(f'./user_data2/log/{args.logfile}')
log.info(f'='*60)
log.info(f'PyTorch YouTubeDNN 召回（最终版）')
log.info(f'参数: mode={mode}, device={device}, recall_k={args.recall_k}')
log.info(f'       embedding_dim={args.embedding_dim}, seq_max_len={args.seq_max_len}')
log.info(f'       use_faiss_gpu={args.use_faiss_gpu}, gpu_id={gpu_id}')
log.info(f'='*60)


# =========================
# 数据构造
# =========================
def gen_data_set(data):
    """滑窗构造训练样本"""
    data = data.sort_values(['user_id', 'click_timestamp']).copy()
    train_set, test_set = [], []

    for uid, hist_df in tqdm(data.groupby('user_id'), desc='Generating dataset'):
        pos_list = hist_df['click_article_id'].tolist()

        if len(pos_list) == 1:
            train_set.append((uid, [pos_list[0]], pos_list[0], 1, 1))
            test_set.append((uid, [pos_list[0]], pos_list[0], 1, 1))
            continue

        for i in range(1, len(pos_list)):
            hist = pos_list[:i][::-1]
            target = pos_list[i]
            if i != len(pos_list) - 1:
                train_set.append((uid, hist, target, 1, len(hist)))
            else:
                test_set.append((uid, hist, target, 1, len(hist)))

    random.shuffle(train_set)
    random.shuffle(test_set)
    return train_set, test_set


def pad_sequences_np(sequences, maxlen, value=0):
    arr = np.full((len(sequences), maxlen), value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        trunc = seq[:maxlen]
        arr[i, :len(trunc)] = trunc
    return arr


def gen_model_input(samples, seq_max_len):
    uid = np.array([x[0] for x in samples], dtype=np.int64)
    seq = [x[1] for x in samples]
    iid = np.array([x[2] for x in samples], dtype=np.int64)
    label = np.array([x[3] for x in samples], dtype=np.float32)
    hist_len = np.array([min(x[4], seq_max_len) for x in samples], dtype=np.int64)

    seq_pad = pad_sequences_np(seq, maxlen=seq_max_len, value=0)

    model_input = {
        'user_id': uid,
        'hist_article_id': seq_pad,
        'hist_len': hist_len,
        'click_article_id': iid
    }
    return model_input, label


class YouTubeDataset(Dataset):
    def __init__(self, model_input, labels):
        self.user_id = torch.LongTensor(model_input['user_id'])
        self.hist_article_id = torch.LongTensor(model_input['hist_article_id'])
        self.hist_len = torch.LongTensor(model_input['hist_len'])
        self.click_article_id = torch.LongTensor(model_input['click_article_id'])
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_id[idx],
            'hist_article_id': self.hist_article_id[idx],
            'hist_len': self.hist_len[idx],
            'click_article_id': self.click_article_id[idx],
            'label': self.labels[idx]
        }


# =========================
# 模型
# =========================
class YouTubeDNNModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        self.user_dnn = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def attention_pooling(self, user_emb, hist_emb, mask):
        scores = torch.sum(hist_emb * user_emb.unsqueeze(1), dim=-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        hist_vec = torch.sum(hist_emb * weights.unsqueeze(-1), dim=1)
        return hist_vec

    def get_user_embedding(self, user_id, hist_article_id):
        user_emb = self.user_embedding(user_id)
        hist_emb = self.item_embedding(hist_article_id)
        mask = (hist_article_id != 0).float()
        hist_vec = self.attention_pooling(user_emb, hist_emb, mask)
        user_vec = torch.cat([user_emb, hist_vec], dim=-1)
        user_vec = self.user_dnn(user_vec)
        return F.normalize(user_vec, dim=1)

    def get_item_embedding(self, item_id):
        item_emb = self.item_embedding(item_id)
        return F.normalize(item_emb, dim=1)

    def forward(self, user_id, hist_article_id, target_item):
        user_vec = self.get_user_embedding(user_id, hist_article_id)
        item_vec = self.get_item_embedding(target_item)
        logits = torch.matmul(user_vec, item_vec.t())
        labels = torch.arange(user_vec.size(0), device=user_vec.device)
        return logits, labels


# =========================
# 训练
# =========================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc='Training'):
        user_id = batch['user_id'].to(device)
        hist_article_id = batch['hist_article_id'].to(device)
        click_article_id = batch['click_article_id'].to(device)

        optimizer.zero_grad()
        logits, labels = model(user_id, hist_article_id, click_article_id)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def build_user_hist_for_query(df_click_encoded, df_query_encoded, seq_max_len):
    """给query用户构造历史序列"""
    user_hist = (
        df_click_encoded.sort_values(['user_id', 'click_timestamp'])
        .groupby('user_id')['click_article_id']
        .agg(list)
        .to_dict()
    )

    rows = []
    for uid in df_query_encoded['user_id'].unique():
        hist = user_hist.get(uid, [])
        hist = hist[::-1][:seq_max_len]
        rows.append((uid, hist))

    out = pd.DataFrame(rows, columns=['user_id', 'hist'])
    out['hist_len'] = out['hist'].apply(len)
    out['hist_pad'] = list(pad_sequences_np(out['hist'].tolist(), maxlen=seq_max_len, value=0))
    return out


@torch.no_grad()
def extract_query_user_embeddings(model, query_hist_df, batch_size, device):
    """批量提取用户向量"""
    model.eval()
    all_embs, all_uids = [], []

    uids = query_hist_df['user_id'].values
    hists = np.stack(query_hist_df['hist_pad'].values)

    n = len(query_hist_df)
    for i in tqdm(range(0, n, batch_size), desc='Extracting user embeddings'):
        j = min(i + batch_size, n)
        b_uid = torch.LongTensor(uids[i:j]).to(device)
        b_hist = torch.LongTensor(hists[i:j]).to(device)

        emb = model.get_user_embedding(b_uid, b_hist).cpu().numpy()
        all_embs.append(emb)
        all_uids.extend(uids[i:j].tolist())

    return np.vstack(all_embs), all_uids


@torch.no_grad()
def extract_item_embeddings(model, item_ids_encoded, batch_size, device):
    """批量提取物品向量"""
    model.eval()
    item_ids_encoded = np.array(item_ids_encoded, dtype=np.int64)

    all_embs = []
    n = len(item_ids_encoded)
    for i in tqdm(range(0, n, batch_size), desc='Extracting item embeddings'):
        j = min(i + batch_size, n)
        b_item = torch.LongTensor(item_ids_encoded[i:j]).to(device)
        emb = model.get_item_embedding(b_item).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs)


def train_youtubednn(df_click_raw, embedding_dim=32, seq_max_len=30):
    """训练模型"""
    df_click = df_click_raw.copy()

    # 编码
    user_le = LabelEncoder()
    item_le = LabelEncoder()

    df_click['user_id_raw'] = df_click['user_id']
    df_click['click_article_id_raw'] = df_click['click_article_id']

    df_click['user_id'] = user_le.fit_transform(df_click['user_id']) + 1
    df_click['click_article_id'] = item_le.fit_transform(df_click['click_article_id']) + 1

    num_users = int(df_click['user_id'].max()) + 1
    num_items = int(df_click['click_article_id'].max()) + 1

    log.info(f'num_users={num_users}, num_items={num_items}')

    train_set, test_set = gen_data_set(df_click)
    train_input, train_label = gen_model_input(train_set, seq_max_len)

    train_dataset = YouTubeDataset(train_input, train_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = YouTubeDNNModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log.info(f'开始训练 {args.epochs} epochs...')
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        log.info(f'Epoch {ep}/{args.epochs}, loss={loss:.6f}')

    # 保存模型
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_path}/model.pt')
    log.info(f'模型已保存: {model_path}/model.pt')

    # 编码映射
    user_raw_to_enc = {raw: enc for raw, enc in zip(df_click['user_id_raw'], df_click['user_id'])}
    item_raw_to_enc = {raw: enc for raw, enc in zip(df_click['click_article_id_raw'], df_click['click_article_id'])}
    user_enc_to_raw = {v: k for k, v in user_raw_to_enc.items()}
    item_enc_to_raw = {v: k for k, v in item_raw_to_enc.items()}

    return model, df_click, user_raw_to_enc, item_raw_to_enc, user_enc_to_raw, item_enc_to_raw


def build_faiss_index(item_embs, use_gpu=False, gpu_id=0):
    """构建FAISS索引"""
    dim = item_embs.shape[1]
    item_embs = item_embs.astype(np.float32)
    
    # 确保C-contiguous
    if not item_embs.flags['C_CONTIGUOUS']:
        item_embs = np.ascontiguousarray(item_embs)
    
    faiss.normalize_L2(item_embs)
    
    # CPU索引
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(item_embs)
    
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            log.info(f'转移索引到GPU {gpu_id}...')
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)
            log.info('GPU索引构建成功')
            return gpu_index
        except Exception as e:
            log.warning(f'GPU索引失败: {e}，使用CPU')
    
    return cpu_index


def recall_with_faiss(df_query_raw, df_click_raw, df_click_encoded, model,
                      user_raw_to_enc, item_enc_to_raw, seq_max_len=30, recall_k=50):
    """
    召回（带已点击过滤）
    """
    # 预计算：每个用户已点击的文章集合（原始ID）
    log.info('构建用户点击历史映射...')
    user_clicked_raw = (
        df_click_raw.groupby('user_id')['click_article_id']
        .apply(set)
        .to_dict()
    )
    
    # query用户编码
    df_query = df_query_raw.copy()
    df_query['user_id_enc'] = df_query['user_id'].map(user_raw_to_enc)
    
    # 过滤冷启动用户
    cold_users = df_query[df_query['user_id_enc'].isna()]['user_id'].tolist()
    if cold_users:
        log.warning(f'冷启动用户 {len(cold_users)} 个，将被跳过')
        df_query = df_query[df_query['user_id_enc'].notna()].copy()
    
    if len(df_query) == 0:
        log.error('无有效查询用户')
        return pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])
    
    q = df_query[['user_id_enc']].drop_duplicates()
    q.columns = ['user_id']
    
    # 构建历史序列
    query_hist_df = build_user_hist_for_query(df_click_encoded, q, seq_max_len=seq_max_len)
    
    # 提取向量
    log.info('提取用户向量...')
    user_embs, user_ids_enc = extract_query_user_embeddings(model, query_hist_df, batch_size=4096, device=device)
    
    log.info('提取物品向量...')
    item_ids_enc = sorted(list(item_enc_to_raw.keys()))
    item_embs = extract_item_embeddings(model, item_ids_enc, batch_size=4096, device=device)
    
    # 构建FAISS索引
    log.info('构建FAISS索引...')
    index = build_faiss_index(item_embs, use_gpu=args.use_faiss_gpu, gpu_id=gpu_id)
    
    # 检索（多召回一些用于过滤）
    search_k = min(recall_k + 50, len(item_ids_enc))  # 多召回50个，但不超过总数
    log.info(f'FAISS检索: {len(user_embs)}用户 × {search_k}候选')
    
    faiss.normalize_L2(user_embs.astype(np.float32))
    sims, idxs = index.search(user_embs.astype(np.float32), search_k)
    
    # 解码映射
    index_to_item_enc = {i: item_ids_enc[i] for i in range(len(item_ids_enc))}
    user_enc_to_raw = {v: k for k, v in user_raw_to_enc.items()}
    
    # 组装结果（带过滤）
    log.info('组装召回结果（过滤已点击）...')
    rows = []
    query_label_map = dict(zip(df_query_raw['user_id'], df_query_raw['click_article_id']))
    
    for row_i, u_enc in enumerate(tqdm(user_ids_enc, desc='Filtering')):
        u_raw = user_enc_to_raw.get(int(u_enc))
        if u_raw is None:
            continue
        
        target_item_raw = query_label_map.get(u_raw, -1)
        clicked_set = user_clicked_raw.get(u_raw, set())  # 该用户已点击的文章
        
        cand_item_enc_idx = idxs[row_i]
        cand_sims = sims[row_i]
        
        filtered_count = 0
        for idx_pos, (ie, sc) in enumerate(zip(cand_item_enc_idx, cand_sims)):
            if ie < 0:
                continue
            
            item_enc = index_to_item_enc[int(ie)]
            item_raw = item_enc_to_raw[item_enc]
            
            # 关键：过滤已点击
            if item_raw in clicked_set:
                continue
            
            label = np.nan if target_item_raw == -1 else (1 if item_raw == target_item_raw else 0)
            rows.append([u_raw, item_raw, float(sc), label])
            filtered_count += 1
            
            if filtered_count >= recall_k:  # 达到指定数量停止
                break
    
    df_data = pd.DataFrame(rows, columns=['user_id', 'article_id', 'sim_score', 'label'])
    
    if len(df_data) == 0:
        log.warning('召回结果为空')
        return df_data
    
    # 排序
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    
    # 统计过滤效果
    total_cand = len(idxs) * len(idxs[0]) if len(idxs) > 0 else 0
    final_cand = len(df_data)
    log.info(f'召回统计: 原始候选={total_cand}, 过滤后={final_cand}, 过滤率={(1-final_cand/total_cand)*100:.1f}%')
    
    return df_data


if __name__ == '__main__':
    if mode == 'valid':
        df_click_raw = pd.read_pickle('./user_data2/data/offline/click.pkl')
        df_query_raw = pd.read_pickle('./user_data2/data/offline/query.pkl')
        emb_file = './user_data2/data/offline/youtubednn_emb.pkl'
        save_file = './user_data2/data/offline/recall_youtubednn.pkl'
        model_path = './user_data2/model/offline'
    else:
        df_click_raw = pd.read_pickle('./user_data2/data/online/click.pkl')
        df_query_raw = pd.read_pickle('./user_data2/data/online/query.pkl')
        emb_file = './user_data2/data/online/youtubednn_emb.pkl'
        save_file = './user_data2/data/online/recall_youtubednn.pkl'
        model_path = './user_data2/model/online'

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    log.info(f'df_click shape={df_click_raw.shape}, df_query shape={df_query_raw.shape}')

    # 训练
    model, df_click_encoded, user_raw_to_enc, item_raw_to_enc, user_enc_to_raw, item_enc_to_raw = train_youtubednn(
        df_click_raw,
        embedding_dim=args.embedding_dim,
        seq_max_len=args.seq_max_len
    )

    # 保存映射
    with open(emb_file, 'wb') as f:
        pickle.dump({
            'user_raw_to_enc': user_raw_to_enc,
            'item_raw_to_enc': item_raw_to_enc,
            'user_enc_to_raw': user_enc_to_raw,
            'item_enc_to_raw': item_enc_to_raw
        }, f)

    # 召回
    log.info('='*60)
    log.info('开始召回...')
    df_data = recall_with_faiss(
        df_query_raw=df_query_raw,
        df_click_raw=df_click_raw,  # 传入原始点击数据用于过滤
        df_click_encoded=df_click_encoded,
        model=model,
        user_raw_to_enc=user_raw_to_enc,
        item_enc_to_raw=item_enc_to_raw,
        seq_max_len=args.seq_max_len,
        recall_k=args.recall_k
    )

    if len(df_data) == 0:
        log.error('召回失败')
        exit(1)

    log.info(f'召回结果 shape={df_data.shape}')
    log.debug(f'\n{df_data.head()}')

    # 评估
    if mode == 'valid':
        log.info('='*60)
        log.info('评估指标')
        
        total = df_query_raw[df_query_raw['click_article_id'] != -1].user_id.nunique()
        
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total
        )
        
        metrics = [
            f'HitRate@5={hitrate_5:.6f}, MRR@5={mrr_5:.6f}',
            f'HitRate@10={hitrate_10:.6f}, MRR@10={mrr_10:.6f}',
            f'HitRate@20={hitrate_20:.6f}, MRR@20={mrr_20:.6f}',
            f'HitRate@40={hitrate_40:.6f}, MRR@40={mrr_40:.6f}',
            f'HitRate@50={hitrate_50:.6f}, MRR@50={mrr_50:.6f}'
        ]
        for m in metrics:
            log.info(m)

    # 保存
    df_data.to_pickle(save_file)
    log.info(f'召回结果已保存: {save_file}')
    log.info('='*60)
    log.info('程序执行完毕')