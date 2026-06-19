# ============================================================================
# KIỂM RÒ RỈ IN-DOMAIN: chia NGẪU NHIÊN per-flow  vs  GROUP theo source-IP
#   Câu hỏi: con số in-domain ~0.99 có bị thổi do tương quan host (cùng host
#   nằm cả train lẫn test) không? IP KHÔNG phải đặc trưng (đã bỏ), nhưng flow
#   cùng host vẫn tương quan -> chia ngẫu nhiên có thể rò rỉ.
#   Cách: với mỗi bộ, đo in-domain dưới (a) random split, (b) group theo
#   IPV4_SRC_ADDR (StratifiedGroupKFold, không host nào ở cả 2 phía).
#   Nếu grouped << random -> rò rỉ host; nếu ~ nhau -> in-domain cao là THẬT.
#   Tự chứa: tải lại data, giải nén ToN.
# ============================================================================
import subprocess, sys
try: import lightgbm as lgb
except Exception:
    subprocess.run([sys.executable,"-m","pip","install","-q","lightgbm"]); import lightgbm as lgb

import os, glob, zipfile, shutil, gc, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
SEED=42; MAX_ROWS_PER_DS=150_000; N_SEEDS=3; CLIP=1e30
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

def load_with_ip(name,max_rows,chunk=500_000):
    """Trả về X (đặc trưng SỐ, ĐÃ bỏ IP), y, và group=source-IP (để chia nhóm)."""
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
    ipc=next((c for c in df.columns if c.upper() in ['IPV4_SRC_ADDR','SRC IP','SRCIP']), None)
    groups=df[ipc].astype(str).values if ipc else np.arange(len(df))   # nếu không có IP -> mỗi flow 1 nhóm (=random)
    Xdf=df.drop(columns=[c for c in ID_DROP+[la] if c in df.columns],errors='ignore')
    Xdf=Xdf.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-CLIP,CLIP)
    feats=[c for c in Xdf.columns if Xdf[c].std()>0]
    X=to_f32(Xdf[feats].values)
    del df,Xdf; gc.collect()
    return X,y,groups,len(np.unique(groups))

class MLP(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h//2,2))
    def forward(s,x): return s.net(x)

def fit_predict(kind,Xtr,ytr,Xte,seed):
    if kind=='RF':
        m=RandomForestClassifier(n_estimators=120,max_depth=22,class_weight='balanced',n_jobs=-1,random_state=seed)
        m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]
    if kind=='LGBM':
        m=lgb.LGBMClassifier(n_estimators=300,num_leaves=64,learning_rate=0.05,class_weight='balanced',
                             n_jobs=-1,random_state=seed,verbose=-1)
        m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]
    # MLP (scale theo train)
    set_seed(seed); nf=Xtr.shape[1]
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xtr_s=to_f32(sc.fit_transform(Xtr)); Xte_s=to_f32(sc.transform(Xte))
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
    model.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xte_s),16384):
            o.append(torch.softmax(model(torch.tensor(Xte_s[i:i+16384]).to(device)).float(),1)[:,1].cpu().numpy())
    del model; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(o)

def AUROC(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')
def AUPRC(y,p): return average_precision_score(y,p) if len(np.unique(y))>1 else float('nan')

# ------------------------------ RUN ------------------------------
print("\n"+"="*92)
print("IN-DOMAIN: RANDOM split  vs  GROUP-by-source-IP split  (AUROC | AUPRC), mean qua seed")
print("="*92)
print(f"  {'Dataset':<6}{'model':<6}{'rand AUROC':>12}{'grp AUROC':>12}{'ΔAUROC':>9}{'rand AUPRC':>12}{'grp AUPRC':>12}  groups")
print("  "+"-"*86)
for n in DATASETS:
    r=load_with_ip(n,MAX_ROWS_PER_DS)
    if r is None: print(f"  ⚠️ bỏ {SHORT[n]}"); continue
    X,y,groups,ng=r
    for kind in ['RF','LGBM','MLP']:
        ra_ro,ra_pr,gr_ro,gr_pr=[],[],[],[]
        for s in range(N_SEEDS):
            # (a) random
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=SEED+s,stratify=y)
            p=fit_predict(kind,Xtr,ytr,Xte,SEED+s); ra_ro.append(AUROC(yte,p)); ra_pr.append(AUPRC(yte,p))
            # (b) group theo source-IP
            try:
                sgkf=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+s)
                tri,tei=next(sgkf.split(X,y,groups))
                if len(np.unique(y[tei]))<2: raise ValueError("test thiếu lớp")
                p=fit_predict(kind,X[tri],y[tri],X[tei],SEED+s)
                gr_ro.append(AUROC(y[tei],p)); gr_pr.append(AUPRC(y[tei],p))
            except Exception as e:
                gr_ro.append(float('nan')); gr_pr.append(float('nan'))
        ra,gro=np.nanmean(ra_ro),np.nanmean(gr_ro)
        print(f"  {SHORT[n]:<6}{kind:<6}{ra:>12.3f}{gro:>12.3f}{gro-ra:>+9.3f}"
              f"{np.nanmean(ra_pr):>12.3f}{np.nanmean(gr_pr):>12.3f}  {ng:>6}")
    del X,y,groups; gc.collect(); torch.cuda.empty_cache()

print("\nDONE.")
print("Đọc: grouped ≈ random (~0.99) => in-domain cao là THẬT, không do học vẹt host (vì IP đã bỏ khỏi đặc trưng).")
print("     grouped << random => có rò rỉ tương quan host -> in-domain bị thổi (cũng là một phát hiện tốt cho bài).")
print("     'groups' = số source-IP duy nhất; nếu quá ít, group-split kém ổn định (đọc thận trọng).")
