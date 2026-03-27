import pandas as pd
import ast

df = pd.read_csv('C:/endoscapes/all_metadata.csv')
df_kf = df[df['is_ds_keyframe'] == True].copy()
df_kf['cvs_mean'] = df_kf['avg_cvs'].apply(lambda x: sum(ast.literal_eval(x)) / 3)
df_kf['label_binary'] = (df_kf['cvs_mean'] >= 0.5).astype(int)

for split in ['train', 'val', 'test']:
    try:
        with open(f'C:/endoscapes/{split}_vids.txt') as f:
            # Parse as float first, then convert to int
            vids = set(int(float(v.strip())) for v in f.readlines() if v.strip())
        sub = df_kf[df_kf['vid'].isin(vids)]
        pos = sub['label_binary'].sum()
        print(f'{split}: {len(sub)} keyframes | CVS=1: {pos} ({pos/len(sub):.2%}) | CVS=0: {len(sub)-pos}')
    except FileNotFoundError:
        print(f'C:/endoscapes/{split}_vids.txt not found')