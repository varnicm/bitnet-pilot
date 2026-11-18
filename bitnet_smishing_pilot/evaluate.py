#!/usr/bin/env python3
"""Evaluate one resident OpenAI-compatible local llama-server, CPU only.
Run once for each model against the same sample. Requires Python stdlib.
Optional: psutil for process-tree RSS peak, otherwise Linux /proc RSS for main PID.
"""
import argparse,csv,json,time,urllib.request,urllib.error,re,statistics,math,os,threading
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--input',default='sample_300.csv')
p.add_argument('--model',required=True,help='Identifier for saved result files')
p.add_argument('--port',type=int,default=8080)
p.add_argument('--server-pid',type=int,default=0,help='PID of llama-server for memory sampling')
p.add_argument('--output-dir',default='results')
p.add_argument('--smoke',type=int,default=0,help='Run first N rows only; results are debugging, not final')
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--timeout',type=float,default=180)
p.add_argument('--max-tokens',type=int,default=5)
a=p.parse_args()
with open(a.input,newline='',encoding='utf-8') as f: data=list(csv.DictReader(f))
if a.smoke:data=data[:a.smoke]
if not data:raise SystemExit('No data')
Path(a.output_dir).mkdir(exist_ok=True,parents=True)
url=f'http://127.0.0.1:{a.port}/v1/chat/completions'
SYSTEM='You are an SMS security classifier. Classify each SMS as smishing or benign. Reply with exactly one capital letter: A for smishing; B for benign. Do not explain.'
def call(message):
 body={'model':'local','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'SMS: '+message+'\nLabel (A or B):'}],'temperature':0,'max_tokens':a.max_tokens,'stream':False}
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=a.timeout) as resp: result=json.load(resp)
 return str(result['choices'][0]['message'].get('content') or '').strip()
def predict(raw):
 clean=raw.strip().upper()
 if re.fullmatch(r'\s*A\s*[.\s]*',clean):return 'smishing'
 if re.fullmatch(r'\s*B\s*[.\s]*',clean):return 'benign'
 return 'invalid'
try:
 import psutil
 proc=psutil.Process(a.server_pid) if a.server_pid else None
except ImportError:proc=None
peak=[0];stop=threading.Event()
def memory():
 if not a.server_pid:return
 while not stop.is_set():
  try:
   if proc:
    ps=[proc]+proc.children(recursive=True)
    rss=sum(x.memory_info().rss for x in ps if x.is_running())
   else:
    with open(f'/proc/{a.server_pid}/status') as f: s=f.read()
    m=re.search(r'^VmRSS:\s+(\d+)\s+kB',s,re.M)
    rss=int(m.group(1))*1024 if m else 0
   peak[0]=max(peak[0],rss)
  except (OSError,Exception):pass
  stop.wait(.05)
t=threading.Thread(target=memory,daemon=True);t.start()
try:
 for i in range(min(a.warmup,len(data))):call(data[i]['message'])
 output=[]
 for i,r in enumerate(data,1):
  st=time.perf_counter();error=''
  try:raw=call(r['message']);pred=predict(raw)
  except Exception as e:raw='';pred='invalid';error=str(e)
  ms=1000*(time.perf_counter()-st)
  output.append({'id':r['id'],'gold':r['label'],'prediction':pred,'raw_output':raw,'latency_ms':round(ms,3),'error':error})
  if i%20==0:print(f'{a.model}: {i}/{len(data)} complete',flush=True)
finally:stop.set();t.join(timeout=1)
valid=[x for x in output if x['prediction']!='invalid']
tp=sum(x['gold']=='smishing' and x['prediction']=='smishing' for x in output)
fp=sum(x['gold']=='benign' and x['prediction']=='smishing' for x in output)
fn=sum(x['gold']=='smishing' and x['prediction']!='smishing' for x in output)
tn=sum(x['gold']=='benign' and x['prediction']=='benign' for x in output)
n=len(output);precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
lat=sorted(x['latency_ms'] for x in output)
def pct(vals,q):
 if not vals:return None
 pos=(len(vals)-1)*q;lo=int(pos);hi=min(lo+1,len(vals)-1)
 return vals[lo]+(vals[hi]-vals[lo])*(pos-lo)
metrics={'model':a.model,'n':n,'smoke':bool(a.smoke),'tp':tp,'fp':fp,'fn_including_invalid':fn,'tn':tn,'invalid_count':n-len(valid),'accuracy_invalid_wrong':(tp+tn)/n,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'false_positive_rate':fp/sum(x['gold']=='benign' for x in output),'median_latency_ms':statistics.median(lat),'p95_latency_ms':pct(lat,.95),'peak_process_rss_gib':peak[0]/1024**3 if peak[0] else None,'memory_sampling':'psutil_process_tree' if proc else ('linux_proc_main_pid' if a.server_pid else 'not_measured'),'server_port':a.port,'warmup':a.warmup,'max_tokens':a.max_tokens,'notes':'single run; balanced sample; invalid predictions count as incorrect; request roundtrip latency includes prompt processing and generation'}
stem=a.model+('_smoke' if a.smoke else '')
with open(Path(a.output_dir)/(stem+'_predictions.csv'),'w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
(Path(a.output_dir)/(stem+'_metrics.json')).write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
