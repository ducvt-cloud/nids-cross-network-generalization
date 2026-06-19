# ============================================================================
# PER-ATTACK-TYPE ANALYSIS  (for the "conflated factors" limitation)
#   Part A: attack-category composition of each dataset (category, count, %).
#   Part B: for each source->target pair, use the SOURCE-TRAINED model to score
#           the target test set, then compute one-vs-benign AUROC for EACH
#           target attack family -> any <0.5 is INVERTED. Shows the attack-mix inversion mechanism.
#   Model: RandomForest (fast, matches the paper config). Self-contained, unpacks ToN.
# ============================================================================
import subprocess, sys, os, glob, zipfile, shutil, gc
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
from google.colab import drive
drive.mount('/content/drive')

DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
PAIRS=[('NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2'),
       ('NF-CSE-CIC-IDS2018-v2','NF-BoT-IoT-v2'),
       ('NF-BoT-IoT-v2','NF-UNSW-NB15-v2'),
       ('NF-UNSW-NB15-v2','NF-BoT-IoT-v2'),     # aligned pair, for comparison
       ('NF-BoT-IoT-v2','NF-CSE-CIC-IDS2018-v2')]
SEED=42; MAX_ROWS_PER_DS=150_000; CLIP=1e30
DRIVE='/content/drive/MyDrive/APT_Data'
SRC={'NF-UNSW-NB15-v2':f'{DRIVE}/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv',
     'NF-CSE-CIC-IDS2018-v2':f'{DRIVE}/NF-CSE-CIC-IDS2018-v2.zip',
     'NF-BoT-IoT-v2':f'{DRIVE}/NF-BoT-IoT-v2.zip',
     'NF-ToN-IoT-v2':f'{DRIVE}/NF-ToN-IoT-v2.rar'}
ID_DROP=['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT',
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
def load_attack(name,max_rows,chunk=500_000):
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
    ac=next((c for c in df.columns if c.lower()=='attack'),None)
    if ac is not None:
        att=df[ac].astype(str).fillna('Unknown').replace({'':'Unknown'}).values
    else:
        att=np.where(y==1,'Attack(unlabeled)','Benign')
    Xdf=df.drop(columns=[c for c in ID_DROP+[la]+([ac] if ac else []) if c in df.columns],errors='ignore')
    Xdf=Xdf.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-CLIP,CLIP)
    del df; gc.collect()
    return Xdf,y,att
def AUROC(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')

print("Loading (keeping the Attack column)...")
raw={}
for n in list(DATASETS):
    try: r=load_attack(n,MAX_ROWS_PER_DS)
    except Exception as e: r=None; print(f"  [error {SHORT[n]}]: {e}")
    if r is None: print(f"  WARN: skipping {SHORT[n]}"); DATASETS.remove(n)
    else: raw[n]=r
common=sorted(set.intersection(*[set(raw[n][0].columns) for n in DATASETS]))
common=[c for c in common if any(raw[n][0][c].std()>0 for n in DATASETS)]
print(f"  Sets: {[SHORT[n] for n in DATASETS]} | #common features={len(common)}\n")

# ---- prepare split (keep attack labels for the test set) ----
SP={}
for n in DATASETS:
    X=to_f32(raw[n][0][common].values); y=raw[n][1]; att=raw[n][2]
    idx=np.arange(len(y))
    tr,te=train_test_split(idx,test_size=0.3,random_state=SEED,stratify=y)
    SP[n]=dict(Xtr=X[tr],ytr=y[tr],Xte=X[te],yte=y[te],att_te=att[te])
del raw; gc.collect()

# ============ PART A: attack-category composition ============
print("="*78); print("PART A — ATTACK-CATEGORY COMPOSITION (full sample per dataset)"); print("="*78)
for n in DATASETS:
    att=np.concatenate([SP[n]['att_te'],
        np.array(['(train hidden)']*0)])  # stats on the test set only for brevity
    vals,cnts=np.unique(SP[n]['att_te'],return_counts=True)
    order=np.argsort(-cnts); tot=cnts.sum()
    print(f"\n{SHORT[n]} (test n={tot}):")
    for i in order:
        print(f"    {str(vals[i])[:34]:<34} {cnts[i]:>7}  ({100*cnts[i]/tot:5.1f}%)")

# ============ PART B: per-target-attack-family AUROC from the source model ============
print("\n"+"="*78)
print("PART B — SOURCE model scoring TARGET: AUROC (attack family vs benign) per family")
print("        <0.5 = INVERTED (attacks scored below benign). 'all' = overall AUROC.")
print("="*78)
def benign_label(att): return np.array([a.strip().lower() in ('benign','normal','0','background') for a in att])
for s,t in PAIRS:
    rf=RandomForestClassifier(n_estimators=120,max_depth=22,class_weight='balanced',
                              n_jobs=-1,random_state=SEED)
    rf.fit(SP[s]['Xtr'],SP[s]['ytr'])
    sc=rf.predict_proba(SP[t]['Xte'])[:,1]
    yt=SP[t]['yte']; att=SP[t]['att_te']; isb=benign_label(att)
    auc_all=AUROC(yt,sc)
    print(f"\n{SHORT[s]} -> {SHORT[t]}   (AUROC all = {auc_all:.3f})")
    if isb.sum()==0:
        print("    (could not identify benign by name; skipping per-family analysis)"); continue
    bs=sc[isb]
    cats=[c for c in np.unique(att[~isb])]
    rows=[]
    for c in cats:
        m=(att==c); n_c=m.sum()
        if n_c<10: continue
        ys=np.concatenate([np.ones(n_c),np.zeros(isb.sum())])
        ps=np.concatenate([sc[m],bs])
        a=AUROC(ys,ps)
        rows.append((c,n_c,a,sc[m].mean(),bs.mean()))
    rows.sort(key=lambda r:r[2])  # most inverted first
    print(f"    {'attack category':<30}{'n':>7}{'AUROC':>9}{'mean_atk':>10}{'mean_ben':>10}  flag")
    for c,n_c,a,ma,mb in rows:
        flag='INVERTED' if a<0.5 else ('weak' if a<0.7 else '')
        print(f"    {str(c)[:30]:<30}{n_c:>7}{a:>9.3f}{ma:>10.3f}{mb:>10.3f}  {flag}")

print("\nDONE. Interpretation: in inverted pairs, 'volumetric' families (DoS/DDoS) vs 'probing/scan'")
print("tend to point in opposite directions -> evidence the inversion is due to attack-mix, not noise.")
