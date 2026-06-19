# ============================================================================
# METHOD: DANN (second UDA) + FEW-SHOT, with AUROC & AUPRC
#   (A) DANN (domain-adversarial) vs baseline on INVERTED pairs -> test whether "unsupervised
#       UDA is insufficient" holds beyond CORAL alone.
#   (B) Few-shot: AUROC & AUPRC vs k target labels/class (k=0,10,50,100) + ceiling.
#   Self-contained: reloads data, unpacks ToN.
# ============================================================================
import subprocess, sys, os, glob, zipfile, shutil, gc, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

DATASETS=['NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2']
SHORT={'NF-CSE-CIC-IDS2018-v2':'CSE','NF-UNSW-NB15-v2':'UNSW','NF-BoT-IoT-v2':'BoT','NF-ToN-IoT-v2':'ToN'}
DANN_PAIRS=[('NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2'),
            ('NF-CSE-CIC-IDS2018-v2','NF-BoT-IoT-v2'),
            ('NF-BoT-IoT-v2','NF-UNSW-NB15-v2')]
FEW_PAIRS=[('NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2'),
           ('NF-CSE-CIC-IDS2018-v2','NF-BoT-IoT-v2'),
           ('NF-BoT-IoT-v2','NF-UNSW-NB15-v2'),
           ('NF-UNSW-NB15-v2','NF-CSE-CIC-IDS2018-v2'),
           ('NF-UNSW-NB15-v2','NF-BoT-IoT-v2'),
           ('NF-ToN-IoT-v2','NF-CSE-CIC-IDS2018-v2')]
K_LIST=[0,10,50,100]
SEED=42; MAX_ROWS_PER_DS=120_000; N_SEEDS=2; CLIP=1e30
EPOCHS=20; BS=2048; HID=128
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
SP={}
for n in DATASETS:
    X=to_f32(raw[n][0][common].values); y=raw[n][1]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    SP[n]=dict(Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte,base=yte.mean())
del raw; gc.collect()
nf=len(common)
DANN_PAIRS=[(s,t) for s,t in DANN_PAIRS if s in DATASETS and t in DATASETS]
FEW_PAIRS=[(s,t) for s,t in FEW_PAIRS if s in DATASETS and t in DATASETS]

def class_w(y):
    c=np.bincount(y,minlength=2); return torch.FloatTensor(np.clip([np.sqrt(len(y)/max(v,1)) for v in c],1,10)).to(device)

# ---------------- (A) DANN ----------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambd): ctx.l=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.l, None

class DANN(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.feat=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                             nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU())
        s.lab=nn.Linear(h//2,2); s.dom=nn.Sequential(nn.Linear(h//2,64),nn.GELU(),nn.Linear(64,2))
    def forward(s,x,lambd=0.0):
        z=s.feat(x); return s.lab(z), s.dom(GRL.apply(z,lambd))

def train_dann(Xs,ys,Xt,seed):
    set_seed(seed); model=DANN(nf,HID).to(device)
    crit=nn.CrossEntropyLoss(weight=class_w(ys)); dcrit=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys); Xt_t=torch.tensor(Xt)
    for e in range(EPOCHS):
        p=e/max(EPOCHS-1,1); lambd=2.0/(1+np.exp(-10*p))-1
        model.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx].to(device); yb=ys_t[idx].to(device)
            tj=torch.randint(0,len(Xt_t),(len(idx),)); xt=Xt_t[tj].to(device)
            lab_s,dom_s=model(xb,lambd); _,dom_t=model(xt,lambd)
            dl=torch.zeros(len(idx),dtype=torch.long,device=device); tl=torch.ones(len(idx),dtype=torch.long,device=device)
            loss=crit(lab_s,yb)+dcrit(dom_s,dl)+dcrit(dom_t,tl)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model

def predict_dann(model,X):
    model.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),16384):
            lg,_=model(torch.tensor(X[i:i+16384]).to(device),0.0); o.append(torch.softmax(lg.float(),1)[:,1].cpu().numpy())
    return np.concatenate(o)

# ---------------- (B) few-shot ----------------
class MLP(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU(),nn.Dropout(0.2),nn.Linear(h//2,2))
    def forward(s,x): return s.net(x)
def train_few(Xs,ys,Xf,yf,seed):
    set_seed(seed); model=MLP(nf,HID).to(device); crit=nn.CrossEntropyLoss(weight=class_w(ys))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys); has=len(Xf)>0
    if has: Xf_t=torch.tensor(Xf); yf_t=torch.tensor(yf)
    for e in range(EPOCHS):
        model.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx]; yb=ys_t[idx]
            if has:
                tj=torch.randint(0,len(Xf_t),(len(idx),)); xb=torch.cat([xb,Xf_t[tj]]); yb=torch.cat([yb,yf_t[tj]])
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model
def predict(model,X):
    model.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),16384):
            o.append(torch.softmax(model(torch.tensor(X[i:i+16384]).to(device)).float(),1)[:,1].cpu().numpy())
    return np.concatenate(o)
def pick(Xtr,ytr,k,seed):
    rng=np.random.RandomState(seed); idxs=[]
    for c in [0,1]:
        ci=np.where(ytr==c)[0]
        if len(ci): idxs.append(rng.choice(ci,min(k,len(ci)),replace=False))
    idx=np.concatenate(idxs) if idxs else np.array([],dtype=int)
    return Xtr[idx],ytr[idx]

# ---------------- RUN (A) DANN ----------------
print("\n"+"="*78+"\n(A) DANN vs BASELINE on INVERTED pairs — AUROC | AUPRC (target base-rate)\n"+"="*78)
for s,t in DANN_PAIRS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xs_s=to_f32(sc.fit_transform(SP[s]['Xtr'])); ys=SP[s]['ytr']
    Xt_tr=to_f32(sc.transform(SP[t]['Xtr'])); Xt_te=to_f32(sc.transform(SP[t]['Xte'])); yt=SP[t]['yte']
    bro,bpr,dro,dpr=[],[],[],[]
    for sd in range(N_SEEDS):
        mb=train_few(Xs_s,ys,np.empty((0,nf),np.float32),np.empty(0,np.int64),SEED+sd)
        pb=predict(mb,Xt_te); bro.append(AUROC(yt,pb)); bpr.append(AUPRC(yt,pb)); del mb
        md=train_dann(Xs_s,ys,Xt_tr,SEED+sd); pd_=predict_dann(md,Xt_te); dro.append(AUROC(yt,pd_)); dpr.append(AUPRC(yt,pd_)); del md
        gc.collect(); torch.cuda.empty_cache()
    print(f"  {SHORT[s]:>4}->{SHORT[t]:<4} (base={SP[t]['base']:.3f}) | "
          f"base AUROC={np.mean(bro):.3f} AUPRC={np.mean(bpr):.3f} | "
          f"DANN AUROC={np.mean(dro):.3f} AUPRC={np.mean(dpr):.3f}")

# ---------------- RUN (B) few-shot ----------------
print("\n"+"="*78+"\n(B) FEW-SHOT — AUROC / AUPRC vs k (target labels/class)\n"+"="*78)
ceil={}
for t in DATASETS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xin=to_f32(sc.fit_transform(SP[t]['Xtr'])); Xte=to_f32(sc.transform(SP[t]['Xte']))
    m=train_few(Xin,SP[t]['ytr'],np.empty((0,nf),np.float32),np.empty(0,np.int64),SEED)
    ceil[t]=(AUROC(SP[t]['yte'],predict(m,Xte)),AUPRC(SP[t]['yte'],predict(m,Xte))); del m; gc.collect()
for s,t in FEW_PAIRS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xs_s=to_f32(sc.fit_transform(SP[s]['Xtr'])); ys=SP[s]['ytr']
    Xt_tr=to_f32(sc.transform(SP[t]['Xtr'])); Xt_te=to_f32(sc.transform(SP[t]['Xte'])); yt=SP[t]['yte']
    line=f"  {SHORT[s]:>4}->{SHORT[t]:<5}(base={SP[t]['base']:.3f}) "
    for k in K_LIST:
        ro,pr=[],[]
        for sd in range(N_SEEDS):
            if k==0: Xf,yf=np.empty((0,nf),np.float32),np.empty(0,np.int64)
            else: Xf,yf=pick(Xt_tr,SP[t]['ytr'],k,SEED+sd)
            m=train_few(Xs_s,ys,Xf,yf,SEED+sd); p=predict(m,Xt_te); ro.append(AUROC(yt,p)); pr.append(AUPRC(yt,p)); del m
            gc.collect(); torch.cuda.empty_cache()
        line+=f" k{k}:{np.mean(ro):.2f}/{np.mean(pr):.2f}"
    line+=f" | ceil {ceil[t][0]:.2f}/{ceil[t][1]:.2f}"
    print(line)

print("\nDONE.  (format: AUROC/AUPRC)")
print("(A) If DANN also fails to lift inverted pairs (like CORAL) => 'unsupervised UDA is insufficient' is robust (not CORAL-only).")
print("(B) See how large k must be for AUROC&AUPRC to cross a useful threshold; read AUPRC with the target base-rate.")
