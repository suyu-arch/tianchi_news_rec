# Tianchi News Recommendation

天池新闻推荐项目复现与改进版。

本项目以天池新闻推荐赛 Top2 思路为基础进行复现，并在此基础上继续扩展了多路召回、排序模型和融合策略，当前版本重点包含：

- 复现 Top2 方案中的经典召回与排序链路
- 新增 `swing` 召回
- 新增 `YouTubeDNN` 召回（PyTorch 实现）
- 新增基于文章 embedding 的 `cold start` 召回
- 新增 `LightGBM Ranker` 排序
- 新增 `DIN` 排序
- 新增加权融合与 LR stacking 融合

## 1. 方法总览

当前主流程如下：

1. `data.py`
   - `valid` 模式下，从训练点击日志中抽样一部分用户构造验证集
   - `online` 模式下，直接使用测试集用户构造待预测 query

2. 多路召回
   - `recall_itemCF.py`
   - `recall_binework.py`
   - `recall_Word2Vec.py`
   - `recall_YoutubeDNN_pytorch.py`
   - `recall_swing.py`
   - `recall_cold_start.py`

3. 召回融合
   - `recall7.py` 对多路召回结果做用户内归一化、加权合并和截断
   - 当前代码默认融合 6 路：`itemcf + binetwork + w2v + youtubednn + swing + cold_start`

4. 排序特征
   - `rank_feature3.py` 在召回结果上增加统计特征、时间特征、相似度特征、交叉特征

5. 排序模型
   - `rank_lgbm.py`
   - `rank_lgbm_ranker.py`
   - `rank_DIN2.py`

6. 融合
   - `rank_fusion_lr.py`
   - 支持加权融合、LR stacking
   - `online` 模式下默认使用离线搜索得到的经验权重做融合

## 2. 数据准备

把比赛原始数据放到 `data/` 目录下：

- `articles.csv`
- `articles_emb.csv`
- `train_click_log.csv`
- `testA_click_log.csv`

数据说明文件 `data/DataA1121.md` 中给出的官方链接如下：

- `articles.csv`: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/articles.csv
- `articles_emb.csv`: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/articles_emb.csv
- `train_click_log.csv`: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/train_click_log.csv
- `testA_click_log.csv`: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/testA_click_log.csv

## 3. 环境依赖

建议使用 Python 3.9 或 3.10。

核心依赖包括：

- `numpy`
- `pandas`
- `scikit-learn`
- `lightgbm`
- `gensim`
- `faiss-cpu` 或兼容环境下的 `faiss-gpu`
- `torch`
- `deepctr-torch`
- `joblib`
- `tqdm`
- `multitasking`

一个可参考的安装命令如下：

```bash
pip install numpy pandas scikit-learn lightgbm gensim faiss-cpu torch joblib tqdm multitasking deepctr-torch
```

如果你使用 GPU，请根据本机 CUDA 版本安装对应的 `torch`。

## 4. 运行方式

### 4.1 一键离线验证

```bash
python run.py
```

离线流程会依次执行：

```text
data
-> emb_i2i_sim
-> itemCF recall
-> bi-network recall
-> word2vec recall
-> YouTubeDNN recall
-> swing recall
-> cold start recall
-> multi-recall fusion
-> feature engineering
-> lgbm ranking
-> lgbm ranker
-> DIN ranking
-> fusion / stacking
```

### 4.2 一键线上预测

```bash
python run_online.py
```

### 4.3 分步执行

如果你想单独调试某一层，也可以按下面顺序逐步运行：

```bash
python data.py --mode valid --logfile debug.log
python embedding_sim.py --mode valid --save_path ./user_data2/sim/offline/emb_i2i_sim.pkl --logfile debug.log
python recall_itemCF.py --mode valid --logfile debug.log
python recall_binework.py --mode valid --logfile debug.log
python recall_Word2Vec.py --mode valid --logfile debug.log
python recall_YoutubeDNN_pytorch.py --mode valid --logfile debug.log
python recall_swing.py --mode valid --logfile debug.log
python recall_cold_start.py --mode valid --logfile debug.log
python recall7.py --mode valid --logfile debug.log
python rank_feature3.py --mode valid --logfile debug.log
python rank_lgbm.py --mode valid --logfile debug.log
python rank_lgbm_ranker.py --mode valid --logfile debug.log
python rank_DIN2.py --mode valid --logfile debug.log
python rank_fusion_lr.py --mode valid --logfile debug.log
```

## 5. 主要产物

常见输出路径如下：

- 召回融合结果：`./user_data2/data/offline/recall7.pkl`
- 排序特征：`./user_data2/data/offline/feature3.pkl`
- LightGBM OOF：`./user_data2/prediction_result/oof_lgbm_valid.csv`
- LightGBM Ranker OOF：`./user_data2/prediction_result/oof_lgbm_ranker_valid.csv`
- DIN OOF：`./user_data2/prediction_result/oof_din_valid3.csv`
- 融合结果：`./user_data2/prediction_result/result_valid_stacking_weighted.csv`
- 线上提交文件：`./user_data2/prediction_result/result_online_stacking_weighted.csv`

## 6. 致谢

- 阿里天池新闻推荐赛
- 原赛题 Top2 思路
- 在此基础上的个人复现、扩展与实验记录
