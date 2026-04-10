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
parser = argparse.ArgumentParser(description='PyTorch YouTubeDNN 召回（Sampled Softmax版）')
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
parser.add_argument('--num_neg_samples', type=int, default=1000, 
                    help='Sampled Softmax的负采样数量（论文默认1000-10000）')
parser.add_argument('--use_freq_sampling', action='store_true', default=True,
                    help='使用频率感知的负采样（解决长尾问题）')
parser.add_argument('--sampling_power', type=float, default=0.75,
                    help='频率采样的幂次（论文推荐0.75）')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Sampled Softmax温度系数（默认1.0）')
parser.add_argument('--exclude_pos_from_neg', action='store_true', default=True,
                    help='负采样时排除正样本（避免冲突）')

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
log.info(f'PyTorch YouTubeDNN 召回（Sampled Softmax版）')
log.info(f'参数: mode={mode}, device={device}, recall_k={args.recall_k}')
log.info(f'       embedding_dim={args.embedding_dim}, seq_max_len={args.seq_max_len}')
log.info(f'       num_neg_samples={args.num_neg_samples}, use_freq_sampling={args.use_freq_sampling}')
log.info(f'       sampling_power={args.sampling_power}, temperature={args.temperature}')
log.info(f'       exclude_pos_from_neg={args.exclude_pos_from_neg}')
log.info(f'='*60)


# =========================
# 数据构造
# =========================
def gen_data_set(data):
    """滑窗构造训练样本"""
    data = data.sort_values(['user_id', 'click_timestamp']).copy()
    train_set, test_set = [], []

    for uid, hist_df in tqdm(data.groupby('user_id'), desc='Generating dataset'):
        pos_list = hist_df['click_article_id'].tolist()#没个用户的点击序列

        if len(pos_list) == 1:#如果用户只有一个点击，那么训练集和测试集都用这个样本，标签为1，历史序列长度为1
            train_set.append((uid, [pos_list[0]], pos_list[0], 1, 1))
            test_set.append((uid, [pos_list[0]], pos_list[0], 1, 1))
            continue

        for i in range(1, len(pos_list)):
            hist = pos_list[:i][::-1]#将历史序列反转
            target = pos_list[i]
            if i != len(pos_list) - 1:#如果不是最后一个样本，那么训练集用这个样本，标签为1，历史序列长度为i；如果是最后一个样本，那么测试集用这个样本，标签为1，历史序列长度为i
                train_set.append((uid, hist, target, 1, len(hist)))
            else:
                test_set.append((uid, hist, target, 1, len(hist)))

    random.shuffle(train_set)#打乱训练集和测试集的顺序，避免模型过拟合
    random.shuffle(test_set)
    return train_set, test_set


def pad_sequences_np(sequences, maxlen, value=0):#sequences是一个列表，里面是每个用户的历史点击序列，maxlen是历史序列的最大长度，value是padding的值
    #将历史序列进行padding，长度为maxlen，padding的值为value。超过maxlen的部分会被截断
    arr = np.full((len(sequences), maxlen), value, dtype=np.int64)#创建一个全是value的二维数组，行数为序列的数量，列数为maxlen
    for i, seq in enumerate(sequences):#
        trunc = seq[:maxlen]
        arr[i, :len(trunc)] = trunc
    return arr


def gen_model_input(samples, seq_max_len):#samples是一个列表，里面是每个用户的训练样本，格式为(uid, hist, target, label, hist_len)，seq_max_len是历史序列的最大长度
    '''将样本转换为模型输入格式'''
    uid = np.array([x[0] for x in samples], dtype=np.int64)
    #每个样本的第一个元素是用户ID，第二个元素是历史点击序列，第三个元素是目标物品ID，第四个元素是标签，第五个元素是历史序列长度
    seq = [x[1] for x in samples]
    iid = np.array([x[2] for x in samples], dtype=np.int64)
    label = np.array([x[3] for x in samples], dtype=np.float32)
    hist_len = np.array([min(x[4], seq_max_len) for x in samples], dtype=np.int64)

    seq_pad = pad_sequences_np(seq, maxlen=seq_max_len, value=0)

    model_input = {
        'user_id': uid,
        'hist_article_id': seq_pad,
        'hist_len': hist_len,
        'click_article_id': iid#目标物品ID
    }
    return model_input, label


class YouTubeDataset(Dataset):#定义一个PyTorch数据集类，继承自torch.utils.data.Dataset
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


# ========================= 频率感知采样器（优化版） =========================
class FrequencyAwareSampler:
    """
    YouTubeDNN论文中的频率感知负采样
    解决长尾物品难以被采样到的问题
    """
    def __init__(self, item_freq, num_items, power=0.75):#item_freq是一个字典，key是物品ID，value是物品的出现频率；num_items是物品总数（包括padding）；power是平滑因子
        """
        item_freq: 每个物品的出现频率（1-indexed，0是padding）
        power: 平滑因子0.75
        """
        self.num_items = num_items#物品总数（包括padding）
        self.power = power
        
        # 计算采样概率（跳过index 0，因为是padding）
        freq_array = np.zeros(num_items, dtype=np.float32)
        for item_id, freq in item_freq.items():
            if 0 < item_id < num_items:
                freq_array[item_id] = freq ** power
        
        # 归一化
        total = freq_array.sum()
        if total > 0:
            self.sampling_probs = freq_array / total#采样概率是频率的power次方除以总和，越频繁的物品被采样的概率越大
        else:
            self.sampling_probs = np.ones(num_items) / num_items#如果没有频率信息，使用均匀分布采样
        
        # 预计算alias table（O(1)采样）
        self._build_alias_table()
        
        # 缓存概率张量用于快速查询
        self.sampling_probs_tensor = torch.FloatTensor(self.sampling_probs)
        
    def _build_alias_table(self):
        """构建Alias Table实现O(1)采样"""
        n = self.num_items
        probs = self.sampling_probs * n#将采样概率缩放到[0, n)区间
        
        small = []#存储概率小于1的索引
        large = []#存储概率大于等于1的索引
        
        for i in range(n):
            if probs[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        self.alias = np.zeros(n, dtype=np.int32)#alias table存储别名索引
        self.prob = np.zeros(n, dtype=np.float32)#prob table存储概率值
        
        while small and large:
            l = small.pop()#取一个小矮子
            g = large.pop()#取一个大个子
            
            self.prob[l] = probs[l]# 第 l 个桶，原主占 probs[l] 这么高
            self.alias[l] = g# 剩下的高度，由大高个 g 来填充
            
            probs[g] = probs[g] + probs[l] - 1.0# 关键：大高个被削掉了一块，更新它的剩余概率
            
            if probs[g] < 1.0:#如果大高个现在也变矮了，放回small，否则继续放回large
                small.append(g)
            else:
                large.append(g)
        
        while large:
            self.prob[large.pop()] = 1.0#剩下的大个子，概率都设为1
        
        while small:
            self.prob[small.pop()] = 1.0
    
    def sample(self, batch_size, num_samples, exclude_ids=None):
        """
        批量采样负样本，支持排除指定ID（避免采到正样本）
        
        Args:
            batch_size: 批次大小
            num_samples: 每个样本采多少负样本
            exclude_ids: [batch_size] 需要排除的ID（通常是正样本）
        
        Returns: [batch_size, num_samples] 的负样本item_id
        """
        if exclude_ids is None or not args.exclude_pos_from_neg:#如果没有指定排除ID，或者配置不要求排除，那么直接使用标准采样（可能会采到正样本，但概率较低）
            # 标准采样（可能采到正样本，但概率低）
            return self._sample_simple(batch_size, num_samples)
        
        # 排除正样本的采样（避免冲突）
        return self._sample_exclusive(batch_size, num_samples, exclude_ids)
    
    def _sample_simple(self, batch_size, num_samples):#num_samples是每个样本需要采多少负样本
        """标准Alias Table采样"""
        idx = np.random.randint(0, self.num_items, size=(batch_size, num_samples))
        #随机生成一个(batch_size, num_samples)的索引矩阵，值在[0, num_items)范围内
        coin = np.random.random((batch_size, num_samples))#随机生成一个(batch_size, num_samples)的概率矩阵，值在[0, 1)范围内
        mask = coin < self.prob[idx]#根据概率表决定是否使用原索引还是别名索引
        result = np.where(mask, idx, self.alias[idx])
        return torch.LongTensor(result)
    
    def _sample_exclusive(self, batch_size, num_samples, exclude_ids):
        """
        采样时排除指定ID（避免负样本与正样本冲突）
        使用拒绝采样策略
        """
        result = np.zeros((batch_size, num_samples), dtype=np.int64)
        #batch_size该批次中样本数量
        for i in range(batch_size):
            exclude = {exclude_ids[i].item()}#需要排除的ID集合，通常是正样本ID
            samples = []
            max_attempts = num_samples * 10  # 防止无限循环
            
            attempts = 0
            while len(samples) < num_samples and attempts < max_attempts:
                # 先盲采一批候选者
                batch_needed = num_samples - len(samples)
                candidates = self._sample_simple(1, batch_needed * 2)[0].numpy()#采样更多的候选者，增加成功率
                
                for c in candidates:
                    # 如果采到的不是正样本(exclude)且不是填充位(0)，才收下
                    if c not in exclude and c != 0:  # 排除padding
                        samples.append(c)
                        if len(samples) >= num_samples:
                            break
                attempts += batch_needed
            
            # 如果不够，用均匀随机填充（fallback）
            while len(samples) < num_samples:
                c = np.random.randint(1, self.num_items)
                if c not in exclude:
                    samples.append(c)
            
            result[i] = samples[:num_samples]
        
        return torch.LongTensor(result)
    
    def get_sampling_prob(self, item_ids):
        """
        获取指定物品的采样概率（用于logQ校正）
        支持向量化查询（优化版）
        
        Args:
            item_ids: 可以是 [batch] 或 [batch, num_neg] 形状
        """
        # 统一处理为1D，查询后再reshape
        original_shape = item_ids.shape#
        flat_ids = item_ids.cpu().numpy().reshape(-1)#将输入的item_ids展平为1D数组，方便查询采样概率
        
        # 向量化查询（避免循环）
        probs = self.sampling_probs[flat_ids]
        
        # 转回tensor并恢复形状
        probs_tensor = torch.FloatTensor(probs).to(item_ids.device)
        if len(original_shape) > 1:
            probs_tensor = probs_tensor.view(original_shape)
        
        return probs_tensor


# =========================
# 模型（Sampled Softmax版）
# =========================
class YouTubeDNNModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, 
                 num_neg_samples=1000, use_freq_sampling=True, sampling_power=0.75,
                 temperature=1.0):
        super().__init__()
        self.num_users = num_users#用户总数
        self.num_items = num_items#物品总数（包括padding）
        self.embedding_dim = embedding_dim
        self.num_neg_samples = num_neg_samples
        self.temperature = temperature
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)#论文中使用的是W2V预训练的物品向量，这里为了简化直接随机初始化，实际应用中可以替换为预训练向量
        #采用了 Weight Tying（权重共享） 或者简化的做法,

        # 注意力投影层
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)

        self.user_dnn = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # 频率感知采样器
        self.neg_sampler = None
        self.use_freq_sampling = use_freq_sampling
        self.sampling_power = sampling_power
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)#随机初始化用户嵌入矩阵，均值为0，标准差为0.01
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def init_neg_sampler(self, item_freq):
        """初始化负采样器"""
        if self.use_freq_sampling:#如果配置了使用频率感知采样，那么根据物品频率初始化采样器
            self.neg_sampler = FrequencyAwareSampler(
                item_freq, self.num_items, self.sampling_power
            )
            log.info(f'初始化频率感知采样器: power={self.sampling_power}')
        else:
            log.info('使用均匀随机采样')

    def attention_pooling(self, hist_emb, mask):
        """
        修正版：用历史平均作为Query（标准YouTubeDNN做法）
        """
        # 计算历史平均（带mask）
        mask_expanded = mask.unsqueeze(-1).float()#将mask从[batch, seq_len]扩展为[batch, seq_len, 1]，方便后续广播
        masked_hist = hist_emb * mask_expanded#将历史嵌入矩阵与mask相乘，得到被mask掉的历史嵌入（padding位置会被置零）
        sum_hist = torch.sum(masked_hist, dim=1)#对历史嵌入矩阵在序列维度上求和，得到每个用户的历史嵌入总和
        count = torch.sum(mask, dim=1, keepdim=True).clamp(min=1)
        query = sum_hist / count
        
        # 投影到注意力空间
        query = self.query_proj(query)
        keys = self.key_proj(hist_emb)
        
        # 计算注意力分数
        scores = torch.sum(keys * query.unsqueeze(1), dim=-1)#计算每个历史嵌入与查询的点积，得到注意力分数，形状为[batch, seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)#将padding位置的分数置为一个很小的值，确保softmax后权重接近于0
        weights = torch.softmax(scores, dim=1)#对注意力分数进行softmax，得到权重，形状为[batch, seq_len]
        
        hist_vec = torch.sum(hist_emb * weights.unsqueeze(-1), dim=1)
        return hist_vec

    def get_user_embedding(self, user_id, hist_article_id):
        user_emb = self.user_embedding(user_id)
        hist_emb = self.item_embedding(hist_article_id)
        mask = (hist_article_id != 0)
        
        hist_vec = self.attention_pooling(hist_emb, mask)
        
        user_vec = torch.cat([user_emb, hist_vec], dim=-1)
        user_vec = self.user_dnn(user_vec)
        return F.normalize(user_vec, dim=1)#对用户向量进行L2归一化，确保向量长度为1，便于后续的内积计算和FAISS索引

    def get_item_embedding(self, item_id):
        item_emb = self.item_embedding(item_id)
        return F.normalize(item_emb, dim=1)

    def forward(self, user_id, hist_article_id, target_item, is_training=True):
        """
        Sampled Softmax前向传播（YouTubeDNN论文实现）
        """
        # 获取用户向量
        user_vec = self.get_user_embedding(user_id, hist_article_id)
        
        if not is_training:
            return user_vec#如果不是训练模式，直接返回用户向量，供FAISS索引使用
        
        batch_size = user_vec.size(0)
        
        # 获取正样本向量
        pos_item_vec = self.get_item_embedding(target_item)
        
        # ========================= Sampled Softmax核心 =========================
        # 1. 采样负样本（支持排除正样本）
        if self.neg_sampler is not None and self.training:
            neg_item_ids = self.neg_sampler.sample(
                batch_size, self.num_neg_samples, exclude_ids=target_item
            ).to(user_vec.device)
        else:
            # 均匀随机采样
            neg_item_ids = torch.randint(
                1, self.num_items, 
                (batch_size, self.num_neg_samples),
                device=user_vec.device
            )
        
        # 2. 获取负样本向量
        neg_item_vec = self.get_item_embedding(neg_item_ids.view(-1)).view(
            batch_size, self.num_neg_samples, -1
        )
        
        # 3. 计算logits（带温度系数）
        pos_logits = torch.sum(user_vec * pos_item_vec, dim=1, keepdim=True) / self.temperature
        
        neg_logits = torch.bmm(
            user_vec.unsqueeze(1),
            neg_item_vec.transpose(1, 2)
        ).squeeze(1) / self.temperature
        
        # 4. LogQ校正（向量化优化版）
        if self.neg_sampler is not None and self.use_freq_sampling:
            # 向量化查询概率（避免循环）
            pos_q = self.neg_sampler.get_sampling_prob(target_item).to(user_vec.device)
            neg_q = self.neg_sampler.get_sampling_prob(neg_item_ids).to(user_vec.device)
            
            # 校正
            pos_logits = pos_logits - torch.log(pos_q + 1e-10).unsqueeze(1)
            neg_logits = neg_logits - torch.log(neg_q + 1e-10)
        
        # 5. 合并logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        # 6. 标签：正样本永远是第0个
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_vec.device)
        
        return logits, labels


# =========================
# 训练
# =========================
def train_epoch(model, dataloader, optimizer, device):#训练一个epoch，model是模型实例，dataloader是训练数据加载器，optimizer是优化器，device是计算设备
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training'):
        user_id = batch['user_id'].to(device)
        hist_article_id = batch['hist_article_id'].to(device)
        click_article_id = batch['click_article_id'].to(device)

        optimizer.zero_grad()
        
        logits, labels = model(
            user_id, 
            hist_article_id, 
            click_article_id, 
            is_training=True
        )
        
        loss = F.cross_entropy(logits, labels)#计算交叉熵损失，logits是模型输出的预测值，labels是实际标签（正样本为0，负样本为1），损失值越小表示模型预测越准确
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#梯度裁剪，防止梯度爆炸，max_norm是梯度的最大范数，超过这个值的梯度会被缩放
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


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
        hist = hist[::-1][:seq_max_len]#将历史序列反转，取最近的seq_max_len个点击
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

    uids = query_hist_df['user_id'].values#获取query用户的ID列表
    hists = np.stack(query_hist_df['hist_pad'].values)#获取query用户的历史序列列表，并进行padding，得到一个二维数组，行数是用户数量，列数是历史序列的最大长度

    n = len(query_hist_df)#用户数量
    for i in tqdm(range(0, n, batch_size), desc='Extracting user embeddings'):#分批次提取用户向量，避免一次性处理过多数据导致内存不足
        j = min(i + batch_size, n)#计算当前批次的结束索引，确保不超过总用户数量
        b_uid = torch.LongTensor(uids[i:j]).to(device)#将当前批次的用户ID转换为LongTensor，并移动到计算设备上
        b_hist = torch.LongTensor(hists[i:j]).to(device)#将当前批次的历史序列转换为LongTensor，并移动到计算设备上

        emb = model(b_uid, b_hist, None, is_training=False).cpu().numpy()
        all_embs.append(emb)
        all_uids.extend(uids[i:j].tolist())

    return np.vstack(all_embs), all_uids


@torch.no_grad()
def extract_item_embeddings(model, item_ids_encoded, batch_size, device):
    """批量提取物品向量"""#此时是已经训练好的
    model.eval()
    item_ids_encoded = np.array(item_ids_encoded, dtype=np.int64)#将物品ID列表转换为NumPy数组，确保数据类型为int64，方便后续处理

    all_embs = []
    n = len(item_ids_encoded)
    for i in tqdm(range(0, n, batch_size), desc='Extracting item embeddings'):#分批次提取物品向量，避免一次性处理过多数据导致内存不足
        j = min(i + batch_size, n)
        b_item = torch.LongTensor(item_ids_encoded[i:j]).to(device)#将当前批次的物品ID转换为LongTensor，并移动到计算设备上
        emb = model.get_item_embedding(b_item).cpu().numpy()#调用模型的get_item_embedding方法获取物品向量，返回一个二维数组，行数是当前批次的物品数量，列数是嵌入维度
        all_embs.append(emb)
    return np.vstack(all_embs)


def train_youtubednn(df_click_raw, embedding_dim=32, seq_max_len=30, 
                     num_neg_samples=1000, use_freq_sampling=True, sampling_power=0.75,
                     temperature=1.0):
    """训练模型（Sampled Softmax版）"""
    df_click = df_click_raw.copy()

    # 编码
    user_le = LabelEncoder()
    item_le = LabelEncoder()

    df_click['user_id_raw'] = df_click['user_id']
    df_click['click_article_id_raw'] = df_click['click_article_id']

    df_click['user_id'] = user_le.fit_transform(df_click['user_id']) + 1#将用户ID进行标签编码，得到连续的整数ID，+1是为了保留0作为padding ID
    df_click['click_article_id'] = item_le.fit_transform(df_click['click_article_id']) + 1

    num_users = int(df_click['user_id'].max()) + 1#
    num_items = int(df_click['click_article_id'].max()) + 1#

    log.info(f'num_users={num_users}, num_items={num_items}')

    # 计算物品频率
    item_freq = df_click['click_article_id'].value_counts().to_dict()
    log.info(f'物品频率统计: 平均频率={np.mean(list(item_freq.values())):.2f}, '
             f'最高频={max(item_freq.values())}, 最低频={min(item_freq.values())}')

    train_set, test_set = gen_data_set(df_click)
    train_input, train_label = gen_model_input(train_set, seq_max_len)

    train_dataset = YouTubeDataset(train_input, train_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 初始化模型
    model = YouTubeDNNModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_neg_samples=num_neg_samples,
        use_freq_sampling=use_freq_sampling,
        sampling_power=sampling_power,
        temperature=temperature
    ).to(device)
    
    model.init_neg_sampler(item_freq)#根据物品频率初始化负采样器，确保采样时考虑物品的流行程度，解决长尾问题

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log.info(f'开始训练 {args.epochs} epochs...')
    log.info(f'Sampled Softmax配置: num_neg_samples={num_neg_samples}, '
             f'use_freq_sampling={use_freq_sampling}, temperature={temperature}')
    
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
    item_enc_to_raw = {v: k for v in item_raw_to_enc.items()}

    return model, df_click, user_raw_to_enc, item_raw_to_enc, user_enc_to_raw, item_enc_to_raw


def build_faiss_index(item_embs, use_gpu=False, gpu_id=0):
    """构建FAISS索引"""
    dim = item_embs.shape[1]
    item_embs = item_embs.astype(np.float32)
    
    if not item_embs.flags['C_CONTIGUOUS']:
        item_embs = np.ascontiguousarray(item_embs)
    
    faiss.normalize_L2(item_embs)
    
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
    """召回（带已点击过滤）"""
    log.info('构建用户点击历史映射...')
    user_clicked_raw = (
        df_click_raw.groupby('user_id')['click_article_id']
        .apply(set)
        .to_dict()
    )
    
    df_query = df_query_raw.copy()
    df_query['user_id_enc'] = df_query['user_id'].map(user_raw_to_enc)
    
    # 冷启动用户处理：用热门推荐兜底
    cold_users = df_query[df_query['user_id_enc'].isna()]['user_id'].tolist()
    if cold_users:
        log.warning(f'冷启动用户 {len(cold_users)} 个，将使用热门兜底')
        # 记录冷启动用户，后续用热门填充
        df_query_cold = df_query[df_query['user_id_enc'].isna()].copy()
        df_query = df_query[df_query['user_id_enc'].notna()].copy()
    else:
        df_query_cold = pd.DataFrame()
    
    if len(df_query) == 0 and len(df_query_cold) == 0:
        log.error('无有效查询用户')
        return pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])
    
    results = []
    
    # 正常用户召回
    if len(df_query) > 0:
        q = df_query[['user_id_enc']].drop_duplicates()
        q.columns = ['user_id']
        
        query_hist_df = build_user_hist_for_query(df_click_encoded, q, seq_max_len=seq_max_len)
        
        log.info('提取用户向量...')
        user_embs, user_ids_enc = extract_query_user_embeddings(model, query_hist_df, batch_size=4096, device=device)
        
        log.info('提取物品向量...')
        item_ids_enc = sorted(list(item_enc_to_raw.keys()))
        item_embs = extract_item_embeddings(model, item_ids_enc, batch_size=4096, device=device)
        
        log.info('构建FAISS索引...')
        index = build_faiss_index(item_embs, use_gpu=args.use_faiss_gpu, gpu_id=gpu_id)
        
        search_k = min(recall_k + 50, len(item_ids_enc))#为了过滤掉已点击的物品，实际搜索时多取一些候选，增加过滤后的剩余数量，避免召回结果过少
        log.info(f'FAISS检索: {len(user_embs)}用户 × {search_k}候选')
        
        faiss.normalize_L2(user_embs.astype(np.float32))#对用户向量进行L2归一化，确保与索引中的物品向量在同一空间进行内积计算，得到相似度分数
        sims, idxs = index.search(user_embs.astype(np.float32), search_k)#使用FAISS索引进行近邻搜索，得到每个用户的候选物品索引和相似度分数，sims是相似度分数矩阵，idxs是对应的物品索引矩阵，行数是用户数量，列数是搜索到的候选数量
        
        index_to_item_enc = {i: item_ids_enc[i] for i in range(len(item_ids_enc))}
        user_enc_to_raw = {v: k for k, v in user_raw_to_enc.items()}
        
        log.info('组装召回结果（过滤已点击）...')
        query_label_map = dict(zip(df_query_raw['user_id'], df_query_raw['click_article_id']))
        
        for row_i, u_enc in enumerate(tqdm(user_ids_enc, desc='Filtering')):
            u_raw = user_enc_to_raw.get(int(u_enc))
            if u_raw is None:
                continue
            
            target_item_raw = query_label_map.get(u_raw, -1)
            clicked_set = user_clicked_raw.get(u_raw, set())
            
            cand_item_enc_idx = idxs[row_i]
            cand_sims = sims[row_i]
            
            filtered_count = 0
            for idx_pos, (ie, sc) in enumerate(zip(cand_item_enc_idx, cand_sims)):
                if ie < 0:
                    continue
                
                item_enc = index_to_item_enc[int(ie)]
                item_raw = item_enc_to_raw[item_enc]
                
                if item_raw in clicked_set:
                    continue
                
                label = np.nan if target_item_raw == -1 else (1 if item_raw == target_item_raw else 0)
                results.append([u_raw, item_raw, float(sc), label])
                filtered_count += 1
                
                if filtered_count >= recall_k:
                    break
    
    # 冷启动用户：用全局热门兜底
    if len(df_query_cold) > 0:
        log.info(f'冷启动用户用热门兜底: {len(df_query_cold)}个')
        # 计算全局热门
        popular_items = df_click_raw['click_article_id'].value_counts().head(recall_k).index.tolist()
        
        for _, row in df_query_cold.iterrows():
            u_raw = row['user_id']
            target_item_raw = row.get('click_article_id', -1)
            
            for rank, item_raw in enumerate(popular_items):
                # 冷启动用户分数递减
                sim_score = 1.0 - rank * 0.01
                label = np.nan if target_item_raw == -1 else (1 if item_raw == target_item_raw else 0)
                results.append([u_raw, item_raw, sim_score, label])
                if rank + 1 >= recall_k:
                    break
    
    df_data = pd.DataFrame(results, columns=['user_id', 'article_id', 'sim_score', 'label'])
    
    if len(df_data) == 0:
        log.warning('召回结果为空')
        return df_data
    
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    
    total_cand = len(df_query) * search_k if len(df_query) > 0 else 0
    final_cand = len(df_data)
    log.info(f'召回统计: 原始候选={total_cand}, 最终={final_cand}')
    
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
        seq_max_len=args.seq_max_len,
        num_neg_samples=args.num_neg_samples,
        use_freq_sampling=args.use_freq_sampling,
        sampling_power=args.sampling_power,
        temperature=args.temperature
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
        df_click_raw=df_click_raw,
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