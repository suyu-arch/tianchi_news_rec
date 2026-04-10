# tianchi_news_rec

Tianchi News Recommendation project snapshot.

## Structure

- `code/`: main scripts for data prep, recall, ranking, and pipeline runs
- `data/DataA1121.md`: dataset download references

## Notes

- Large raw datasets and generated artifacts are intentionally excluded from this GitHub snapshot.
- Please download the Tianchi dataset files yourself according to `data/DataA1121.md`.
- Intermediate files such as recall caches, trained models, logs, and prediction outputs are not included.

## Main pipeline

- `code/run.py`: offline pipeline
- `code/run_online.py`: online / submission pipeline

## Core modules

- `code/data.py`
- `code/embedding_sim.py`
- `code/recall_itemCF.py`
- `code/recall_swing.py`
- `code/recall_Word2Vec.py`
- `code/recall_YoutubeDNN_pytorch.py`
- `code/recall_cold_start.py`
- `code/recall.py`
- `code/rank_feature.py`
- `code/rank_feature_baseline.py`
- `code/rank_lgbm_ranker.py`
- `code/utils.py`
