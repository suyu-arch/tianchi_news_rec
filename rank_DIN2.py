import argparse
import gc
import os
import random
import warnings
import pickle
from bisect import bisect_left

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DIN

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings("ignore")

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =========================
# 参数
# =========================
parser = argparse.ArgumentParser(description="DIN for ranking (deepctr-torch, cuda)")
parser.add_argument("--mode", default="valid", choices=["valid", "online"])
parser.add_argument("--logfile", default="din.log")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--hist_max_len", type=int, default=50)
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug_n_users", type=int, default=0, help="只取前N个用户调试，0表示全部")
args, unknown = parser.parse_known_args()

mode = args.mode
logfile = args.logfile
epochs = args.epochs
batch_size = args.batch_size
hist_max_len = args.hist_max_len
num_folds = args.num_folds
num_workers = args.num_workers
debug_n_users = args.debug_n_users

os.makedirs("./user_data2/log", exist_ok=True)
os.makedirs("./user_data2/model", exist_ok=True)
os.makedirs("./user_data2/prediction_result", exist_ok=True)
os.makedirs("./user_data2/encoder", exist_ok=True)

log = Logger(f"./user_data2/log/{logfile}").logger
log.info(
    f"DIN start, mode={mode}, epochs={epochs}, batch_size={batch_size}, "
    f"hist_max_len={hist_max_len}, folds={num_folds}, debug_n_users={debug_n_users}"
)


# =========================
# 字段配置（按你的 feature3 结构）
# =========================
ID_COLS = ["user_id", "article_id"]
LABEL_COL = "label"

# 明确指定离散特征（可按你数据再增减）
SPARSE_FEATURES = [
    "user_id",
    "article_id",
    "category_id",
    "same_as_last_category",
    "is_fresh_24h",
    "is_fresh_72h",
    "is_fresh_7d",
]

# 排除列
EXCLUDE_COLS = {
    LABEL_COL,
    "created_at_datetime",
    "click_datetime",
    "hist_article_id",
    "seq_length",
}

ENCODER_PATH = "./user_data2/encoder/din_label_encoders.pkl"
SCALER_PATH = "./user_data2/encoder/din_dense_scaler.pkl"


def reduce_int_safe(s: pd.Series, fillna=0):
    x = pd.to_numeric(s, errors="coerce").fillna(fillna)
    return x.astype(np.int64)


def pad_seq(seq, max_len, pad_val=0):
    if seq is None or len(seq) == 0:
        return [pad_val] * max_len, 0
    seq = seq[-max_len:]
    l = len(seq)
    if l < max_len:
        seq = [pad_val] * (max_len - l) + seq
    return seq, l


def build_user_click_cache(click_df: pd.DataFrame):
    """
    构建用户点击缓存:
    user_click_ts_map[user_id] = [ts1, ts2, ...] (升序)
    user_click_item_map[user_id] = [aid1, aid2, ...] (与ts对齐)
    """
    c = click_df[["user_id", "click_article_id", "click_timestamp"]].copy()
    c["user_id"] = reduce_int_safe(c["user_id"], fillna=-1)
    c["click_article_id"] = reduce_int_safe(c["click_article_id"], fillna=0)
    c["click_timestamp"] = pd.to_numeric(c["click_timestamp"], errors="coerce").fillna(0).astype(np.int64)

    c = c.sort_values(["user_id", "click_timestamp"], ascending=[True, True])

    user_click_ts_map = {}
    user_click_item_map = {}
    for uid, g in c.groupby("user_id", sort=False):
        user_click_ts_map[uid] = g["click_timestamp"].tolist()
        user_click_item_map[uid] = g["click_article_id"].tolist()
    return user_click_ts_map, user_click_item_map


def infer_candidate_ts(df: pd.DataFrame):
    """
    给候选样本估计"样本时点"，优先级：
    click_timestamp > click_datetime > created_at_ts > created_at_datetime > 0
    """
    out = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

    if "click_timestamp" in df.columns:
        x = pd.to_numeric(df["click_timestamp"], errors="coerce").fillna(0).astype(np.int64)
        out = np.where(x > 0, x, out)

    if "click_datetime" in df.columns:
        dt = pd.to_datetime(df["click_datetime"], errors="coerce")
        ts = (dt.astype("int64") // 10**9).fillna(0).astype(np.int64)
        out = np.where((out == 0) & (ts > 0), ts, out)

    if "created_at_ts" in df.columns:
        x = pd.to_numeric(df["created_at_ts"], errors="coerce").fillna(0).astype(np.int64)
        x = np.where(x > 10**12, x // 1000, x)
        out = np.where((out == 0) & (x > 0), x, out)

    if "created_at_datetime" in df.columns:
        dt = pd.to_datetime(df["created_at_datetime"], errors="coerce")
        ts = (dt.astype("int64") // 10**9).fillna(0).astype(np.int64)
        out = np.where((out == 0) & (ts > 0), ts, out)

    return pd.Series(out, index=df.index).astype(np.int64)


def build_hist_for_samples(df: pd.DataFrame, user_click_ts_map, user_click_item_map, max_len=50):
    """
    对每条样本按样本时点截断历史：
    hist = clicks(user) where click_ts < sample_ts
    """
    candidate_ts = infer_candidate_ts(df)
    user_ids = reduce_int_safe(df["user_id"], fillna=-1).values

    hist_list = []
    seq_len_list = []

    for uid, ts in zip(user_ids, candidate_ts.values):
        ts_list = user_click_ts_map.get(uid, [])
        it_list = user_click_item_map.get(uid, [])

        if len(ts_list) == 0:
            seq, l = pad_seq([], max_len=max_len, pad_val=0)
        else:
            pos = bisect_left(ts_list, ts)
            hist_items = it_list[:pos]
            seq, l = pad_seq(hist_items, max_len=max_len, pad_val=0)

        hist_list.append(seq)
        seq_len_list.append(l)

    df = df.copy()
    df["hist_article_id"] = hist_list
    df["seq_length"] = np.array(seq_len_list, dtype=np.int32)
    return df


def pick_dense_features(df: pd.DataFrame, sparse_features):
    dense = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c in sparse_features:
            continue
        if c in ["hist_article_id", "seq_length"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            dense.append(c)
    return dense


def fit_label_encoders(train_df: pd.DataFrame, sparse_features):
    encoders = {}
    for col in sparse_features:
        le = LabelEncoder()
        vals = train_df[col].fillna("__nan__").astype(str).values
        le.fit(vals)
        encoders[col] = le
    return encoders


def transform_with_encoders(df: pd.DataFrame, encoders):
    out = df.copy()
    for col, le in encoders.items():
        cls2id = {k: i for i, k in enumerate(le.classes_)}
        arr = out[col].fillna("__nan__").astype(str).map(cls2id).fillna(0).astype(np.int64)
        out[col] = arr
    return out


def fit_dense_scaler(train_df: pd.DataFrame, dense_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = train_df[dense_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    scaler.fit(x)
    return scaler


def transform_dense(df: pd.DataFrame, dense_features, scaler):
    out = df.copy()
    x = out[dense_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    out[dense_features] = scaler.transform(x)
    return out


def build_feature_columns(df_train, sparse_features, dense_features, hist_max_len, article_vocab_size):
    sparse_cols = []
    for feat in sparse_features:
        vocab_size = int(df_train[feat].max()) + 1
        sparse_cols.append(SparseFeat(feat, vocabulary_size=max(vocab_size, 2), embedding_dim=16))

    item_feat = SparseFeat(
        "article_id",
        vocabulary_size=max(article_vocab_size, 2),
        embedding_dim=16,
        embedding_name="article_id"
    )

    hist_feat = VarLenSparseFeat(
        SparseFeat(
            "hist_article_id",
            vocabulary_size=max(article_vocab_size, 2),
            embedding_dim=16,
            embedding_name="article_id",
        ),
        maxlen=hist_max_len,
        length_name="seq_length",
    )

    dense_cols = [DenseFeat(feat, 1) for feat in dense_features]

    sparse_cols_no_dup = [x for x in sparse_cols if x.name != "article_id"]
    dnn_feature_columns = sparse_cols_no_dup + [item_feat] + [hist_feat] + dense_cols
    return dnn_feature_columns


def make_model_input(df, sparse_features, dense_features):
    x = {}
    for f in sparse_features:
        x[f] = df[f].values.astype(np.int64)
    x["hist_article_id"] = np.array(df["hist_article_id"].tolist(), dtype=np.int64)
    x["seq_length"] = df["seq_length"].values.astype(np.int64)
    for f in dense_features:
        x[f] = df[f].values.astype(np.float32)
    return x


def train_valid(df_feature, df_query, df_click, debug_n_users=0):
    # ========== 调试模式：限制用户数量 ==========
    # ========== 调试模式：限制用户数量 ==========
    if debug_n_users > 0:
        log.info(f"Debug mode: selecting top {debug_n_users} users")
        
        # 先分离训练集和测试集
        train_mask = df_feature[LABEL_COL].notnull()
        train_users = df_feature.loc[train_mask, 'user_id'].unique()
        test_users = df_feature.loc[~train_mask, 'user_id'].unique()
        
        # 取训练集的前 N 用户
        selected_train_users = train_users[:debug_n_users]
        
        # 测试集取相同的用户（确保有数据）+ 补充一些测试集特有用户
        selected_test_users = test_users[:debug_n_users]
        
        # 合并
        selected_users = np.unique(np.concatenate([selected_train_users, selected_test_users]))
        
        df_feature = df_feature[df_feature['user_id'].isin(selected_users)]
        df_query = df_query[df_query['user_id'].isin(selected_users)]
        df_click = df_click[df_click['user_id'].isin(selected_users)]
        
        log.info(f"Debug data: {len(selected_users)} users, {len(df_feature)} samples")
        log.info(f"  Train samples: {train_mask.sum()}")
        log.info(f"  Test samples: {(~train_mask).sum()}")
    
    
    train_mask = df_feature[LABEL_COL].notnull()
    df_train_all = df_feature.loc[train_mask].copy()
    df_test = df_feature.loc[~train_mask].copy()
    df_train_all[LABEL_COL] = pd.to_numeric(df_train_all[LABEL_COL], errors="coerce").fillna(0).astype(int)
    
    del df_feature
    gc.collect()

    user_click_ts_map, user_click_item_map = build_user_click_cache(df_click)
    df_train_all = build_hist_for_samples(df_train_all, user_click_ts_map, user_click_item_map, max_len=hist_max_len)
    df_test = build_hist_for_samples(df_test, user_click_ts_map, user_click_item_map, max_len=hist_max_len)

    for c in SPARSE_FEATURES:
        if c not in df_train_all.columns:
            df_train_all[c] = 0
            df_test[c] = 0

    dense_features = pick_dense_features(df_train_all, SPARSE_FEATURES)
    log.info(f"sparse_features={SPARSE_FEATURES}")
    log.info(f"dense_features_count={len(dense_features)}")

    oof_parts = []
    pred_test_total = np.zeros(len(df_test), dtype=np.float32)
    test_key = df_test[["user_id", "article_id"]].copy().reset_index(drop=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"torch device: {device}")

    gkf = GroupKFold(n_splits=num_folds)

    for fold_id, (trn_idx, val_idx) in enumerate(
        gkf.split(df_train_all, df_train_all[LABEL_COL], groups=df_train_all["user_id"])
    ):
        log.info(f"========== Fold {fold_id + 1}/{num_folds} ==========")
        
        # 关键修复：编码前保存原始 ID
        val_raw_ids = df_train_all.iloc[val_idx][["user_id", "article_id", LABEL_COL]].copy().reset_index(drop=True)
        
        trn_df = df_train_all.iloc[trn_idx].copy().reset_index(drop=True)
        val_df = df_train_all.iloc[val_idx].copy().reset_index(drop=True)
        tst_df = df_test.copy().reset_index(drop=True)

        encoders = fit_label_encoders(trn_df, SPARSE_FEATURES)
        trn_df = transform_with_encoders(trn_df, encoders)
        val_df = transform_with_encoders(val_df, encoders)
        tst_df = transform_with_encoders(tst_df, encoders)

        article_cls2id = {k: i for i, k in enumerate(encoders["article_id"].classes_)}

        def map_hist(seq):
            return [article_cls2id.get(str(v), 0) for v in seq]

        trn_df["hist_article_id"] = trn_df["hist_article_id"].apply(map_hist)
        val_df["hist_article_id"] = val_df["hist_article_id"].apply(map_hist)
        tst_df["hist_article_id"] = tst_df["hist_article_id"].apply(map_hist)

        if len(dense_features) > 0:
            scaler = fit_dense_scaler(trn_df, dense_features)
            trn_df = transform_dense(trn_df, dense_features, scaler)
            val_df = transform_dense(val_df, dense_features, scaler)
            tst_df = transform_dense(tst_df, dense_features, scaler)

        article_vocab_size = int(max(trn_df["article_id"].max(), val_df["article_id"].max(), tst_df["article_id"].max())) + 1

        dnn_feature_columns = build_feature_columns(
            trn_df,
            sparse_features=SPARSE_FEATURES,
            dense_features=dense_features,
            hist_max_len=hist_max_len,
            article_vocab_size=article_vocab_size
        )

        x_trn = make_model_input(trn_df, SPARSE_FEATURES, dense_features)
        y_trn = trn_df[LABEL_COL].values
        x_val = make_model_input(val_df, SPARSE_FEATURES, dense_features)
        y_val = val_df[LABEL_COL].values
        x_tst = make_model_input(tst_df, SPARSE_FEATURES, dense_features)

        model = DIN(
            dnn_feature_columns=dnn_feature_columns,
            history_feature_list=["article_id"],
            dnn_use_bn=False,
            dnn_hidden_units=(256, 128),
            dnn_activation="relu",
            att_hidden_size=(64, 32),
            att_activation="Dice",
            att_weight_normalization=True,
            dnn_dropout=0.1,
            task="binary",
            device=device,
        )
        model.compile("adam", "binary_crossentropy", metrics=["auc"])

        model.fit(
            x=x_trn,
            y=y_trn,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_val, y_val),
        )

        val_pred = model.predict(x_val, batch_size=batch_size).reshape(-1)
        tst_pred = model.predict(x_tst, batch_size=batch_size).reshape(-1)

        log.info(
            f"fold={fold_id} val_pred std={np.std(val_pred):.8f}, "
            f"min={val_pred.min():.6f}, max={val_pred.max():.6f}"
        )

        # 关键修复：用原始 ID 保存 OOF
        df_oof = val_raw_ids.copy()
        df_oof["pred"] = val_pred
        oof_parts.append(df_oof)

        pred_test_total += tst_pred / num_folds

        torch.save(model.state_dict(), f"./user_data2/model/din_fold{fold_id}.pt")
        joblib.dump(encoders, f"./user_data2/encoder/din_enc_fold{fold_id}.pkl")
        if len(dense_features) > 0:
            joblib.dump(scaler, f"./user_data2/encoder/din_scaler_fold{fold_id}.pkl")

        del model, trn_df, val_df, tst_df, x_trn, x_val, x_tst, val_raw_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_enc = fit_label_encoders(df_train_all, SPARSE_FEATURES)
    joblib.dump(full_enc, ENCODER_PATH)
    if len(dense_features) > 0:
        full_train_enc = transform_with_encoders(df_train_all.copy(), full_enc)
        full_scaler = fit_dense_scaler(full_train_enc, dense_features)
        joblib.dump(full_scaler, SCALER_PATH)

    df_oof = pd.concat(oof_parts, axis=0, ignore_index=True)
    df_oof.sort_values(["user_id", "pred"], ascending=[True, False], inplace=True)
    df_oof.to_csv("./user_data2/prediction_result/oof_din_valid3.csv", index=False)

    prediction = test_key.copy()
    prediction["pred"] = pred_test_total
    prediction.to_csv("./user_data2/prediction_result/test_din_valid3.csv", index=False)

    # 详细评估指标
    total = df_query[df_query["click_article_id"] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(df_oof, total)
    
    log.info("=" * 50)
    log.info("OFFLINE EVALUATION METRICS:")
    log.info(f"  Total users: {total}")
    log.info(f"  HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}")
    log.info(f"  HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}")
    log.info(f"  HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}")
    log.info(f"  HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}")
    log.info(f"  HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}")
    log.info("=" * 50)

    df_sub = gen_sub(prediction)
    df_sub.sort_values(["user_id"], inplace=True)
    df_sub.to_csv("./user_data2/prediction_result/result_valid_din3.csv", index=False)
    log.info("valid done.")


def online_predict(df_feature, df_click, debug_n_users=0):
    df_test = df_feature.copy()
    
    # 调试模式
    if debug_n_users > 0:
        log.info(f"Debug mode: selecting top {debug_n_users} users")
        unique_users = df_test['user_id'].unique()
        selected_users = unique_users[:debug_n_users]
        df_test = df_test[df_test['user_id'].isin(selected_users)]
        df_click = df_click[df_click['user_id'].isin(selected_users)]
        log.info(f"Debug data: {len(selected_users)} users, {len(df_test)} samples")

    user_click_ts_map, user_click_item_map = build_user_click_cache(df_click)
    df_test = build_hist_for_samples(df_test, user_click_ts_map, user_click_item_map, max_len=hist_max_len)

    for c in SPARSE_FEATURES:
        if c not in df_test.columns:
            df_test[c] = 0

    dense_features = pick_dense_features(df_test, SPARSE_FEATURES)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prediction = df_test[["user_id", "article_id"]].copy()
    prediction["pred"] = 0.0

    for fold_id in range(num_folds):
        enc_path = f"./user_data2/encoder/din_enc_fold{fold_id}.pkl"
        scaler_path = f"./user_data2/encoder/din_scaler_fold{fold_id}.pkl"
        model_path = f"./user_data2/model/din_fold{fold_id}.pt"

        if not os.path.exists(enc_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"missing fold artifact: {enc_path} or {model_path}")

        encoders = joblib.load(enc_path)
        tst_df = transform_with_encoders(df_test.copy(), encoders)

        article_cls2id = {k: i for i, k in enumerate(encoders["article_id"].classes_)}
        tst_df["hist_article_id"] = tst_df["hist_article_id"].apply(
            lambda seq: [article_cls2id.get(str(v), 0) for v in seq]
        )

        if len(dense_features) > 0 and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            tst_df = transform_dense(tst_df, dense_features, scaler)

        article_vocab_size = int(tst_df["article_id"].max()) + 1
        dnn_feature_columns = build_feature_columns(
            tst_df,
            sparse_features=SPARSE_FEATURES,
            dense_features=dense_features,
            hist_max_len=hist_max_len,
            article_vocab_size=article_vocab_size
        )

        x_tst = make_model_input(tst_df, SPARSE_FEATURES, dense_features)

        model = DIN(
            dnn_feature_columns=dnn_feature_columns,
            history_feature_list=["article_id"],
            dnn_use_bn=False,
            dnn_hidden_units=(256, 128),
            dnn_activation="relu",
            att_hidden_size=(64, 32),
            att_activation="Dice",
            att_weight_normalization=True,
            dnn_dropout=0.1,
            task="binary",
            device=device,
        )
        model.compile("adam", "binary_crossentropy", metrics=["auc"])
        model.load_state_dict(torch.load(model_path, map_location=device))

        fold_pred = model.predict(x_tst, batch_size=batch_size).reshape(-1)
        prediction["pred"] += fold_pred / num_folds

        del model, tst_df, x_tst
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction.to_csv("./user_data2/prediction_result/test_din_online3.csv", index=False)
    df_sub = gen_sub(prediction)
    df_sub.sort_values(["user_id"], inplace=True)
    df_sub.to_csv("./user_data2/prediction_result/result_online_din3.csv", index=False)
    log.info("online done.")


if __name__ == "__main__":
    if mode == "valid":
        df_feature = pd.read_pickle("./user_data2/data/offline/feature3.pkl")
        df_query = pd.read_pickle("./user_data2/data/offline/query.pkl")
        df_click = pd.read_pickle("./user_data2/data/offline/click.pkl")
        train_valid(df_feature, df_query, df_click, debug_n_users=debug_n_users)
    else:
        df_feature = pd.read_pickle("./user_data2/data/online/feature3.pkl")
        df_click = pd.read_pickle("./user_data2/data/online/click.pkl")
        online_predict(df_feature, df_click, debug_n_users=debug_n_users)