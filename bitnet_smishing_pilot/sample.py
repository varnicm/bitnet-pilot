#!/usr/bin/env python3
"""Prepare a deterministic, balanced SMS evaluation sample. Requires standard library only."""
import argparse,csv,random,json,hashlib
from pathlib import Path

def norm(v):
 s=str(v).strip().lower()
 if s in ('1','smishing','smish','phishing','malicious','spam'): return 'smishing'
 if s in ('0','benign','ham','legitimate','normal'): return 'benign'
 raise ValueError(f'Unknown label {v!r}; map labels to smishing/benign before sampling')

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',default='sample_300.csv')
p.add_argument('--text-col',default='Message')
p.add_argument('--label-col',default='Label')
p.add_argument('--id-col',default='ID')
p.add_argument('--per-class',type=int,default=150)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
with open(a.input,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
if not rows: raise SystemExit('No rows in input CSV')
for col in [a.text_col,a.label_col]:
 if col not in rows[0]: raise SystemExit(f'Missing column {col}; found {list(rows[0])}')
by={'smishing':[],'benign':[]}
for i,r in enumerate(rows):
 if not r[a.text_col].strip():continue
 k=norm(r[a.label_col]); by[k].append({'id':r.get(a.id_col,str(i)) or str(i),'message':r[a.text_col], 'label':k})
rng=random.Random(a.seed)
selected=[]
for k in ('smishing','benign'):
 if len(by[k])<a.per_class: raise SystemExit(f'Need {a.per_class} {k}, found {len(by[k])}')
 selected+=rng.sample(by[k],a.per_class)
rng.shuffle(selected)
with open(a.output,'w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['id','message','label']);w.writeheader();w.writerows(selected)
h=hashlib.sha256(Path(a.output).read_bytes()).hexdigest()
print(json.dumps({'rows':len(selected),'seed':a.seed,'sha256':h,'output':a.output},indent=2))
