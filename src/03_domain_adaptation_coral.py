# ============================================================================
# DOMAIN ADAPTATION (Deep CORAL) — testing the label-conditional prediction
#   Self-contained: reloads data, aligns features, unpacks ToN.
#   For each source->target pair: compare AUROC of BASELINE (source-only) vs CORAL
#   (aligns source/target feature covariance, using UNLABELED target X).
#   Prediction to test: since p(y|x) is INVERTED across domains, CORAL (aligning p(x)) will
#   NOT fix inverted pairs, but may help aligned / mildly shifted pairs.
# ============================================================================
import subprocess, sys, os, glob, zipfile, shutil, gc, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ------------------------------ CONFIG ------------------------------
DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
SEED=42; MAX_ROWS_PER_DS=150_000; N_SEEDS=2; CLIP=1e30
EPOCHS=25; BS=2048; HID=128; LAM_CORAL=1.0
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
    try:
        subprocess.run(['apt-get','install','-y','-q','unrar'],capture_output=True)
        r=subprocess.run(['unrar','x','-o+',src,dst+'/'],capture_output=True)
        if glob.glob(os.path.join(dst,'**','*.csv'),recursive=True): return True
    except Exception: pass
    try:
        subprocess.run(['apt-get','install','-y','-q','p7zip-full'],capture_output=True)
        subprocess.run(['7z','x','-y',src,'-o'+dst],capture_output=True)
        if glob.glob(os.path.join(dst,'**','*.csv'),recursive=True): return True
    except Exception: pass
    try:
        subprocess.run([sys.executable,'-m','pip','install','-q','patool'],capture_output=True)
        import patoolib; patoolib.extract_archive(src,outdir=dst,verbosity=-1)
        if glob.glob(os.path.join(dst,'**','*.csv'),recursive=True): return True
    except Exception: pass
    return False

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

def auroc(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')

# ------------------------------ load + align + split ------------------------------
print("\nLoading & aligning features...")
raw={}
for n in list(DATASETS):
    try: r=load_named(n,MAX_ROWS_PER_DS)
    except Exception as e: r=None; print(f"  [error {SHORT[n]}]: {e}")
    if r is None: print(f"  WARN: skipping {SHORT[n]} ({SRC[n]})"); DATASETS.remove(n)
    else: raw[n]=r
common=sorted(set.intersection(*[set(raw[n][0].columns) for n in DATASETS]))
common=[c for c in common if any(raw[n][0][c].std()>0 for n in DATASETS)]
print(f"  Sets: {[SHORT[n] for n in DATASETS]} | #common features={len(common)}")
SP={}
for n in DATASETS:
    X=to_f32(raw[n][0][common].values); y=raw[n][1]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    SP[n]=dict(Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte)
del raw; gc.collect()
nf=len(common)

# ------------------------------ Deep CORAL ------------------------------
class CoralNet(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.feat=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                             nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU())
        s.clf=nn.Linear(h//2,2)
    def forward(s,x): z=s.feat(x); return s.clf(z),z

def coral_loss(zs,zt):
    d=zs.size(1)
    zs=zs-zs.mean(0,keepdim=True); zt=zt-zt.mean(0,keepdim=True)
    cs=(zs.t()@zs)/max(zs.size(0)-1,1); ct=(zt.t()@zt)/max(zt.size(0)-1,1)
    return ((cs-ct)**2).sum()/(4*d*d)

def train_model(Xs,ys,Xt_unlab,seed,use_coral):
    set_seed(seed)
    c=np.bincount(ys,minlength=2); w=torch.FloatTensor(np.clip([np.sqrt(len(ys)/max(v,1)) for v in c],1,10)).to(device)
    model=CoralNet(nf,HID).to(device); crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys)
    Xt_t=torch.tensor(Xt_unlab) if use_coral else None
    for e in range(EPOCHS):
        model.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx].to(device); yb=ys_t[idx].to(device)
            logits,zs=model(xb); loss=crit(logits,yb)
            if use_coral and len(idx)>2:
                tj=torch.randint(0,len(Xt_t),(len(idx),)); xt=Xt_t[tj].to(device)
                _,zt=model(xt); loss=loss+LAM_CORAL*coral_loss(zs,zt)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model

def predict(model,Xs_):
    model.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xs_),16384):
            lg,_=model(torch.tensor(Xs_[i:i+16384]).to(device)); o.append(torch.softmax(lg.float(),1)[:,1].cpu().numpy())
    return np.concatenate(o)

# ------------------------------ RUN ------------------------------
print("\n"+"="*78)
print("BASELINE (source-only)  vs  CORAL  — AUROC on the TARGET test set")
print("="*78)
rows=[]
for s in DATASETS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xs_s=to_f32(sc.fit_transform(SP[s]['Xtr'])); ys=SP[s]['ytr']
    for t in DATASETS:
        if t==s: continue
        Xt_tr=to_f32(sc.transform(SP[t]['Xtr']))      # unlabeled target X (for CORAL alignment)
        Xt_te=to_f32(sc.transform(SP[t]['Xte'])); yt=SP[t]['yte']
        bas,cor=[],[]
        for sd in range(N_SEEDS):
            mb=train_model(Xs_s,ys,None,SEED+sd,use_coral=False); bas.append(auroc(yt,predict(mb,Xt_te))); del mb
            mc=train_model(Xs_s,ys,Xt_tr,SEED+sd,use_coral=True); cor.append(auroc(yt,predict(mc,Xt_te))); del mc
            gc.collect(); torch.cuda.empty_cache()
        b,co=np.mean(bas),np.mean(cor)
        kind = 'ALIGNED' if b>=0.65 else ('INVERTED' if b<0.45 else 'near-random')
        rows.append((s,t,b,co,kind))
        print(f"  {SHORT[s]:>4} -> {SHORT[t]:<4} | base={b:.3f}  coral={co:.3f}  Δ={co-b:+.3f}  [{kind}]")

# summary by pair type
print("\n"+"-"*78)
for k in ['ALIGNED','near-random','INVERTED']:
    sub=[(co-b) for (_,_,b,co,kk) in rows if kk==k]
    if sub: print(f"  Pair {k:<14}: n={len(sub):>2} | mean Δ (CORAL - base) = {np.mean(sub):+.3f}")
print("\nDONE.")
print("Prediction: CORAL ~0 or worse on INVERTED pairs (p(y|x) inverted, aligning p(x) does not help);")
print("         may be positive on aligned/mildly shifted pairs. If so => 'UDA is insufficient, need few-shot/target labels'.")
