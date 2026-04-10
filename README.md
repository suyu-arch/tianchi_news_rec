# 天池新闻推荐项目复现

本仓库用于复现和整理阿里天池新闻推荐项目，整体方案围绕：

- 多路召回
- 召回融合
- 特征工程
- 排序模型

展开，形成一条完整的新闻推荐实验链路。

---

## 1. 仓库结构

```text
tianchi_news_rec/
├─ code/
│  ├─ data.py
│  ├─ embedding_sim.py
│  ├─ recall.py
│  ├─ recall_itemCF.py
│  ├─ recall_swing.py
│  ├─ recall_Word2Vec.py
│  ├─ recall_YoutubeDNN_pytorch.py
│  ├─ recall_cold_start.py
│  ├─ rank_feature.py
│  ├─ rank_feature_baseline.py
│  ├─ rank_lgbm_ranker.py
│  ├─ run.py
│  ├─ run_online.py
│  └─ utils.py
├─ data/
│  └─ DataA1121.md
└─ README.md
```

各模块作用如下：

### `code/data.py`

用于数据预处理与离线/线上数据划分。

### `code/embedding_sim.py`

基于文章 embedding 构建文章相似度索引，用于 embedding 相似召回。

### `code/recall_itemCF.py`

ItemCF 召回。

### `code/recall_swing.py`

Swing 召回。

### `code/recall_Word2Vec.py`

基于 Skip-gram 的 Word2Vec / Item2Vec 召回。

### `code/recall_YoutubeDNN_pytorch.py`

基于 PyTorch 的双塔召回模型。

### `code/recall_cold_start.py`

冷启动召回逻辑。

### `code/recall.py`

多路召回融合。

### `code/rank_feature.py`

排序特征工程。

### `code/rank_feature_baseline.py`

基线版本的排序特征工程。

### `code/rank_lgbm_ranker.py`

基于 LightGBM Ranker 的排序模型。

### `code/run.py`

离线一键运行主流程。

### `code/run_online.py`

线上预测或提交流程入口。

### `code/utils.py`

日志、评估、提交文件生成等公共工具函数。

---

## 2. 实验环境

推荐环境如下：

- 操作系统：Windows / Linux 均可
- Python：3.9 或 3.10
- 主要依赖：
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `lightgbm`
  - `gensim`
  - `faiss-cpu` 或 `faiss-gpu`
  - `torch`
  - `joblib`
  - `tqdm`
  - `multitasking`

可参考安装命令：

```bash
pip install numpy pandas scikit-learn lightgbm gensim faiss-cpu torch joblib tqdm multitasking
```

如果使用 GPU，请根据本机 CUDA 版本安装对应版本的 `torch` 和 `faiss-gpu`。

---

## 3. 数据准备

数据下载链接见：

- [data/DataA1121.md](./data/DataA1121.md)

需要准备的主要文件包括：

- `articles.csv`
- `articles_emb.csv`
- `train_click_log.csv`
- `testA_click_log.csv`

建议将这些文件按代码中的默认路径放置后再运行实验。

---

## 4. 复现流程

## 4.1 离线实验流程

直接运行：

```bash
python code/run.py
```

主流程大致包括：

1. 数据划分
2. embedding 相似度构建
3. 多路召回
4. 多路召回融合
5. 特征工程
6. 排序模型训练与预测

---

## 4.2 线上预测流程

直接运行：

```bash
python code/run_online.py
```

该流程会基于测试集生成推荐结果。

---

## 4.3 分步运行

如果希望单独调试某个模块，可以按以下顺序逐步执行：

```bash
python code/data.py --mode valid --logfile debug.log
python code/embedding_sim.py --mode valid --logfile debug.log
python code/recall_itemCF.py --mode valid --logfile debug.log
python code/recall_swing.py --mode valid --logfile debug.log
python code/recall_Word2Vec.py --mode valid --logfile debug.log
python code/recall_YoutubeDNN_pytorch.py --mode valid --logfile debug.log
python code/recall_cold_start.py --mode valid --logfile debug.log
python code/recall.py --mode valid --logfile debug.log
python code/rank_feature.py --mode valid --logfile debug.log
python code/rank_lgbm_ranker.py --mode valid --logfile debug.log
```

如果做线上预测，将 `valid` 改成对应线上模式即可。

---

## 5. 项目方法概览

本项目的核心思想是：

### 召回层

使用多路召回从不同角度覆盖候选文章：

- ItemCF：基于协同共现关系
- Swing：强化结构性共同兴趣
- Word2Vec：学习文章语义表示
- YouTubeDNN：双塔向量召回
- 冷启动：补充新文章和低曝光文章

### 融合层

对多路召回结果做归一化和加权，提升候选覆盖质量。

### 排序层

构造用户特征、文章特征和用户-文章交互特征，在候选集上进一步排序。

---

## 6. 说明

本仓库适合用于：

- 推荐系统课程项目复现
- 多路召回与排序实验
- 新闻推荐链路理解
- 后续继续扩展特征工程和排序模型

如果你希望在此基础上继续扩展，可以优先从以下方向入手：

1. 补充新的召回策略
2. 扩展排序特征
3. 引入更多排序模型
4. 优化多路融合权重
