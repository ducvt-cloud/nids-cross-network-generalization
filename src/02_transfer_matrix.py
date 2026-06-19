# ============================================================================
# TRANSFER MATRIX (FINAL) — RF+LGBM+MLP, 5 seeds, AUROC + AUPRC + base-rate
#   - Print the AUROC matrix (mean±std) and AUPRC matrix (mean±std) for EACH model.
#   - Print the base-rate (attack%) of each target -> to interpret AUPRC correctly
#     (AUPRC is only meaningful when attacks are the MINORITY; at BoT 99.6% attack the AUPRC
#      is inflated by the base rate, read together).
#   Self-contained: reloads data, unpacks ToN.
# ============================================================================
import subprocess, sys
try: import lightgbm as lgb
except Exception:
    subprocess.run([sys.executable,"-m","pip","install","-q","lightgbm"]); import lightgbm as lgb

import os, glob, zipfile, shutil, gc, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ------------------------------ CONFIG ------------------------------
DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
SEED=42; MAX_ROWS_PER_DS=150_000; N_SEEDS=5; CLIP=1e30
MLP_EPOCHS=15; MLP_BS=4096; MLP_HID=128
DRIVE='/content/drive/MyDrive/APT_Data'
SRC={'NF-UNSW-NB15-v2':f'{DRIVE}/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv',
     'NF-CSE-CIC-IDS2018-v2':f'{DRIVE}/NF-CSE-CIC-IDS2018-v2.zip',
     'NF-BoT-IoT-v2':f'{DRIVE}/NF-BoT-IoT-v2.zip',
     'NF-ToN-IoT-v2':f'{DRIVE}/NF-ToN-IoT-v2.rar'}
ID_DROP=['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT','Attack',
         'Flow ID','Src IP','Dst IP','Src Port','Dst Port','Timestamp','Unnamed: 0']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def to_f32(a):
    a=np.asarray(a,dtype=np.float64); a=np.nan_to_num(a,nan=0.0,posinf=CLIP,neginf=-CLIP)
    return np.clip(a,-CLIP,CLIP).astype(np.float32)

def extract_rar(src,dst):
    os.makedirs(dst,exist_ok=True)
    if glob.glob(os.path.join(dst,'**','*.csv'),recursive=True): return True
    for cmd in (['apt-get','install','-y','-q','unrar'],['apt-get','install','-y','-q','p7zip-full']):
        try: subprocess.run(cmd,capture_output=True)
        except Exception: pass
    for run in (['unrar','x','-o+',src,dst+'/'],['7z','x','-y',src,'-o'+dst]):
        try:
            subprocess.run(run,capture_output=True)
            if glob.glob(os.path.join(dst,'**','*.csv'),recursive=True): return True
        except Exception: pass
    try:
        subprocess.run([sys.executable,'-m','pip','install','-q','patool'],capture_output=True)
        import patoolib; patoolib.extract_archive(src,outdir=dst,verbosity=-1)
        return bool(glob.glob(os.path.join(dst,'**','*.csv'),recursive=True))
    except Exception: return False

def find_csvs(src):
    if src.endswith('.csv'): return [src] if os.path.exists(src) else []
    dst='/content/_x/'+os.path.splitext(os.path.basename(src))[0]; os.makedirs(dst,exist_ok=True)
    if src.endswith('.zip') and os.path.exists(src):
        with zipfile.ZipFile(src) as z: z.extractall(dst)
    elif src.endswith('.rar'):
        if not os.path.exists(src) or not extract_rar(src,dst): return []
    elif os.path.isdir(src):
        for f in glob.glob(os.path.join(src,'**','*.csv'),recursive=True):
            d=os.path.join(dst,os.path.basename(f))
            if not os.path.exists(d): shutil.copy2(f,d)
    return sorted([c for c in glob.glob(os.path.join(dst,'**','*.csv'),recursive=True)
                   if 'feature' not in os.path.basename(c).lower() and os.path.getsize(c)>5000])

def load_named(name,max_rows,chunk=500_000):
    csvs=find_csvs(SRC[name])
    if not csvs: return None
    total=0
    for f in csvs:
        with open(f) as fh: total+=sum(1 for _ in fh)-1
    frac=min(1.0,max_rows/max(total,1)); rng=np.random.RandomState(SEED); parts=[]
    for f in csvs:
        for ck in pd.read_csv(f,low_memory=False,chunksize=chunk):
            if frac<1.0: ck=ck[rng.random(len(ck))<frac]
            if len(ck): parts.append(ck)
    df=pd.concat(parts,ignore_index=True); del parts; gc.collect()
    df.columns=[str(c).strip() for c in df.columns]
    la=next((c for c in df.columns if c.lower()=='label'),None)
    if la is None: la=df.columns[-1]
    y=(df[la].fillna(0).astype(float)!=0).astype(np.int64).values
    Xdf=df.drop(columns=[c for c in ID_DROP+[la] if c in df.columns],errors='ignore')
    Xdf=Xdf.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-CLIP,CLIP)
    del df; gc.collect()
    return Xdf,y

def AUROC(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')
def AUPRC(y,p): return average_precision_score(y,p) if len(np.unique(y))>1 else float('nan')

# ------------------------------ load + align + split ------------------------------
print("\nLoading & aligning features...")
raw={}
for n in list(DATASETS):
    try: r=load_named(n,MAX_ROWS_PER_DS)
    except Exception as e: r=None; print(f"  [error {SHORT[n]}]: {e}")
    if r is None: print(f"  WARN: skipping {SHORT[n]}"); DATASETS.remove(n)
    else: raw[n]=r
common=sorted(set.intersection(*[set(raw[n][0].columns) for n in DATASETS]))
common=[c for c in common if any(raw[n][0][c].std()>0 for n in DATASETS)]
print(f"  Sets: {[SHORT[n] for n in DATASETS]} | #common features={len(common)}")
SP={}; BASE={}
for n in DATASETS:
    X=to_f32(raw[n][0][common].values); y=raw[n][1]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    SP[n]=dict(Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte); BASE[n]=yte.mean()
    print(f"  {SHORT[n]:<5}: train={len(Xtr):,} test={len(Xte):,} attack%(test)={yte.mean()*100:.2f}")
del raw; gc.collect()
nf=len(common)

# ------------------------------ MLP ------------------------------
class MLP(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h//2,2))
    def forward(s,x): return s.net(x)

def mlp_fit_predict(Xtr_s,ytr,tests_s,seed):
    set_seed(seed)
    c=np.bincount(ytr,minlength=2); w=torch.FloatTensor(np.clip([np.sqrt(len(ytr)/max(v,1)) for v in c],1,10)).to(device)
    model=MLP(nf,MLP_HID).to(device); crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt=torch.tensor(Xtr_s); yt=torch.tensor(ytr)
    for e in range(MLP_EPOCHS):
        model.train(); pm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),MLP_BS):
            j=pm[i:i+MLP_BS]; xb,yb=Xt[j].to(device),yt[j].to(device)
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    def P(Xs_):
        model.eval(); o=[]
        with torch.no_grad():
            for i in range(0,len(Xs_),16384):
                o.append(torch.softmax(model(torch.tensor(Xs_[i:i+16384]).to(device)).float(),1)[:,1].cpu().numpy())
        return np.concatenate(o)
    out=[P(t) for t in tests_s]; del model; gc.collect(); torch.cuda.empty_cache()
    return out

def run_family(kind):
    ro={tr:{te:[] for te in DATASETS} for tr in DATASETS}  # auroc
    pr={tr:{te:[] for te in DATASETS} for tr in DATASETS}  # auprc
    for tr in DATASETS:
        Xtr,ytr=SP[tr]['Xtr'],SP[tr]['ytr']
        if kind=='MLP':
            sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
            Xtr_in=to_f32(sc.fit_transform(Xtr)); tests=[to_f32(sc.transform(SP[te]['Xte'])) for te in DATASETS]
        else:
            tests=[SP[te]['Xte'] for te in DATASETS]
        for s in range(N_SEEDS):
            if kind=='RF':
                m=RandomForestClassifier(n_estimators=120,max_depth=22,class_weight='balanced',n_jobs=-1,random_state=SEED+s)
                m.fit(Xtr,ytr); preds=[m.predict_proba(t)[:,1] for t in tests]
            elif kind=='LGBM':
                m=lgb.LGBMClassifier(n_estimators=300,num_leaves=64,learning_rate=0.05,class_weight='balanced',
                                     n_jobs=-1,random_state=SEED+s,verbose=-1)
                m.fit(Xtr,ytr); preds=[m.predict_proba(t)[:,1] for t in tests]
            else:
                preds=mlp_fit_predict(Xtr_in,ytr,tests,SEED+s)
            for k,te in enumerate(DATASETS):
                ro[tr][te].append(AUROC(SP[te]['yte'],preds[k])); pr[tr][te].append(AUPRC(SP[te]['yte'],preds[k]))
        gc.collect()
    agg=lambda M:{tr:{te:(np.mean(M[tr][te]),np.std(M[tr][te])) for te in DATASETS} for tr in DATASETS}
    return agg(ro),agg(pr)

def print_mat(title,res):
    print("\n  "+title)
    print("  train\\test "+"".join(f"{SHORT[n]:>15}" for n in DATASETS))
    for tr in DATASETS:
        row=f"  {SHORT[tr]:<10}"
        for te in DATASETS:
            m,sd=res[tr][te]; row+=f"{m:>7.3f}±{sd:<6.3f}"
        print(row)

# ------------------------------ RUN ------------------------------
print("\nbase-rate (attack% test) per dataset:  "+" | ".join(f"{SHORT[n]}={BASE[n]*100:.1f}%" for n in DATASETS))
for kind in ['RF','LGBM','MLP']:
    print("\n"+"="*88+f"\nMODEL: {kind}  | {N_SEEDS} seeds (rows=train, cols=test)\n"+"="*88)
    ro,pr=run_family(kind)
    print_mat("AUROC (mean±std):",ro)
    print_mat("AUPRC (mean±std):",pr)

print("\nDONE.")
print("Read: AUROC = primary metric (fair cross-domain comparison). Read AUPRC WITH the base-rate:")
print("  - interpret only when attacks are the minority (CSE 12%, UNSW 4%, ToN 64%).")
print("  - at BoT (99.6% attack) the no-skill AUPRC is ~0.996 -> do NOT use AUPRC for the BoT column.")
print("  - when comparing AUPRC across datasets, remember the random floor = that dataset's base rate.")
