import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

scripts = [
    ('data.py', ['--mode', 'online']),
    ('embedding_sim.py', ['--mode', 'test', '--save_path', './user_data2/sim/online/emb_i2i_sim.pkl']),
    ('recall_itemCF.py', ['--mode', 'online']),
    ('recall_binework.py', ['--mode', 'online']),
    ('recall_Word2Vec.py', ['--mode', 'online']),
    ('recall_YoutubeDNN_pytorch.py', ['--mode', 'online']),
    ('recall_swing.py', ['--mode', 'online']),
    ('recall_cold_start.py', ['--mode', 'online']),
    ('recall7.py', ['--mode', 'online']),
    ('rank_feature3.py', ['--mode', 'online']),
    ('rank_lgbm.py', ['--mode', 'online']),
    ('rank_lgbm_ranker.py', ['--mode', 'online']),
    ('rank_DIN2.py', ['--mode', 'online']),
    ('rank_fusion_lr.py', ['--mode', 'online']),
]

logfile = f"online-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log"

print(f'working dir: {os.getcwd()}')
print(f'logfile: {logfile}\n')

for i, (script, args) in enumerate(scripts, 1):
    print(f'[{i}/{len(scripts)}] running {script}...')

    if not os.path.exists(script):
        print(f'  skip: {script} not found')
        continue

    cmd = [sys.executable, script] + args + ['--logfile', logfile]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'  failed: {script}')
        sys.exit(1)

    print('  done')

print('\nonline prediction pipeline completed')
