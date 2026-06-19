# ============================================================================
# LARGE-SCALE CONFIRMATION  (does the collapse hold at large scale?)
#   Re-run the 4x4 AUROC TRANSFER MATRIX at ~1,000,000 rows/dataset (~7x larger than 150K),
#   with LightGBM (light on RAM, fast on large data). 3 seeds (different splits) -> mean±std.
#   Compared directly with Table 3 (150K) to rule out a "subsampling artifact".
#   Self-contained: reloads, unpacks ToN. Adjust MAX_ROWS_PER_DS if RAM allows.
# ============================================================================
import subprocess, sys, os, glob, zipfile, shutil, gc
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
except Exception:
    subprocess.run([sys.executable,'-m','pip','install','-q','lightgbm'],capture_output=True)
    import lightgbm as lgb
from google.colab import drive
drive.mount('/content/drive')

DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
SEED=42; N_SEEDS=3; MAX_ROWS_PER_DS=1_000_000; CLIP=1e30
DRIVE='/content/drive/MyDrive/APT_Data'
SRC={'NF-UNSW-NB15-v2':f'{DRIVE}/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv',
     'NF-CSE-CIC-IDS2018-v2':f'{DRIVE}/NF-CSE-CIC-IDS2018-v2.zip',
     'NF-BoT-IoT-v2':f'{DRIVE}/NF-BoT-IoT-v2.zip',
     'NF-ToN-IoT-v2':f'{DRIVE}/NF-ToN-IoT-v2.rar'}
ID_DROP=['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT','Attack',
         'Flow ID','Src IP','Dst IP','Src Port','Dst Port','Timestamp','Unnamed: 0']

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

print(f"Loading pool up to {MAX_ROWS_PER_DS:,} rows/dataset (may take a few minutes)...")
POOL={}
for n in list(DATASETS):
    try: r=load_named(n,MAX_ROWS_PER_DS)
    except Exception as e: r=None; print(f"  [error {SHORT[n]}]: {e}")
    if r is None: print(f"  WARN: skipping {SHORT[n]}"); DATASETS.remove(n)
    else:
        POOL[n]=r; print(f"  {SHORT[n]}: {len(r[1]):,} rows | attack-rate={r[1].mean():.3f}")
common=sorted(set.intersection(*[set(POOL[n][0].columns) for n in DATASETS]))
common=[c for c in common if any(POOL[n][0][c].std()>0 for n in DATASETS)]
print(f"  Sets: {[SHORT[n] for n in DATASETS]} | #common features={len(common)}")
Xall={n:to_f32(POOL[n][0][common].values) for n in DATASETS}
yall={n:POOL[n][1] for n in DATASETS}
base={n:yall[n].mean() for n in DATASETS}
del POOL; gc.collect()

def fit_lgbm(X,y,seed):
    spw=(y==0).sum()/max((y==1).sum(),1)
    m=lgb.LGBMClassifier(n_estimators=300,num_leaves=64,learning_rate=0.05,
                         scale_pos_weight=spw,random_state=seed,n_jobs=-1,verbose=-1)
    m.fit(X,y); return m

# AUROC (and AUPRC) matrix, mean±std over N_SEEDS
import numpy as np
AUR={(s,t):[] for s in DATASETS for t in DATASETS}
APR={(s,t):[] for s in DATASETS for t in DATASETS}
for sd in range(N_SEEDS):
    seed=SEED+sd; print(f"\n--- seed {seed} ---")
    spl={}
    for n in DATASETS:
        Xtr,Xte,ytr,yte=train_test_split(Xall[n],yall[n],test_size=0.3,random_state=seed,stratify=yall[n])
        spl[n]=(Xtr,ytr,Xte,yte)
    for s in DATASETS:
        m=fit_lgbm(spl[s][0],spl[s][1],seed)
        for t in DATASETS:
            p=m.predict_proba(spl[t][2])[:,1]
            AUR[(s,t)].append(AUROC(spl[t][3],p)); APR[(s,t)].append(AUPRC(spl[t][3],p))
        print(f"  {SHORT[s]:>4} trained & scored 4 targets")
        del m; gc.collect()
    del spl; gc.collect()

def ms(v): a=np.array(v,float); return np.nanmean(a),np.nanstd(a)
print("\n"+"="*78)
print(f"LARGE-SCALE AUROC MATRIX (~{MAX_ROWS_PER_DS:,}/dataset, LGBM, {N_SEEDS} seeds) — mean±std")
print("rows=train(source), cols=test(target). Compare with Table 3 (150K).")
print("="*78)
hdr="train\\test  "+ "".join(f"{SHORT[t]:>16}" for t in DATASETS); print(hdr)
for s in DATASETS:
    row=f"{SHORT[s]:<11}"
    for t in DATASETS:
        m_,sd_=ms(AUR[(s,t)]); row+=f"  {m_:.3f}±{sd_:.3f}"
    print(row)
print(f"\ntarget base-rate: "+" ".join(f"{SHORT[t]}={base[t]:.3f}" for t in DATASETS))
print("\nAUPRC MATRIX (mean) — read with the base-rate; drop the BoT column (artifact):")
hdr="train\\test  "+ "".join(f"{SHORT[t]:>10}" for t in DATASETS); print(hdr)
for s in DATASETS:
    row=f"{SHORT[s]:<11}"
    for t in DATASETS:
        m_,_=ms(APR[(s,t)]); row+=f"  {m_:>8.3f}"
    print(row)
print("\nDONE. If the off-diagonal still collapses/inverts like Table 3 -> the conclusion holds at scale.")
print("To approach 'full', increase MAX_ROWS_PER_DS (watch RAM); BoT/CSE are very large.")
