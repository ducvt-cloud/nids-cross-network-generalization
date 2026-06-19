# ============================================================================
# ADAPTATION with 5 SEEDS + STD  (for §5.4: CORAL, DANN, few-shot)
#   Part A: baseline vs CORAL vs DANN — AUROC mean±std + Δ, on representative pairs.
#   Part B: few-shot — AUROC/AUPRC mean±std vs k={0,10,50,100} + ceiling.
#   Print results INCREMENTALLY per pair -> if it disconnects, partial results are kept.
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
# 6 representative pairs: 3 inverted + 1 near-random + 1 aligned + 1 CORAL-helped
PAIRS=[('NF-CSE-CIC-IDS2018-v2','NF-UNSW-NB15-v2'),
       ('NF-CSE-CIC-IDS2018-v2','NF-BoT-IoT-v2'),
       ('NF-BoT-IoT-v2','NF-UNSW-NB15-v2'),
       ('NF-UNSW-NB15-v2','NF-CSE-CIC-IDS2018-v2'),
       ('NF-UNSW-NB15-v2','NF-BoT-IoT-v2'),
       ('NF-ToN-IoT-v2','NF-CSE-CIC-IDS2018-v2')]
K_LIST=[0,10,50,100]
SEED=42; N_SEEDS=5; MAX_ROWS_PER_DS=80_000; CLIP=1e30
EPOCHS=15; BS=2048; HID=128; LAM_CORAL=1.0
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
PAIRS=[(s,t) for s,t in PAIRS if s in DATASETS and t in DATASETS]

def class_w(y):
    c=np.bincount(y,minlength=2); return torch.FloatTensor(np.clip([np.sqrt(len(y)/max(v,1)) for v in c],1,10)).to(device)

# ---------- models ----------
class MLP(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU(),nn.Dropout(0.2),nn.Linear(h//2,2))
    def forward(s,x): return s.net(x)
class CoralNet(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.feat=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                             nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU())
        s.clf=nn.Linear(h//2,2)
    def forward(s,x): z=s.feat(x); return s.clf(z),z
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l): ctx.l=l; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.l, None
class DANN(nn.Module):
    def __init__(s,nf,h):
        super().__init__()
        s.feat=nn.Sequential(nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(0.2),
                             nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.GELU())
        s.lab=nn.Linear(h//2,2); s.dom=nn.Sequential(nn.Linear(h//2,64),nn.GELU(),nn.Linear(64,2))
    def forward(s,x,l=0.0): z=s.feat(x); return s.lab(z), s.dom(GRL.apply(z,l))

def coral_loss(zs,zt):
    d=zs.size(1); zs=zs-zs.mean(0,keepdim=True); zt=zt-zt.mean(0,keepdim=True)
    cs=(zs.t()@zs)/max(zs.size(0)-1,1); ct=(zt.t()@zt)/max(zt.size(0)-1,1)
    return ((cs-ct)**2).sum()/(4*d*d)

def train_mlp(Xs,ys,Xf,yf,seed):     # baseline (Xf empty) or few-shot (mix in target labels)
    set_seed(seed); m=MLP(nf,HID).to(device); crit=nn.CrossEntropyLoss(weight=class_w(ys))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys); has=len(Xf)>0
    if has: Xf_t=torch.tensor(Xf); yf_t=torch.tensor(yf)
    for e in range(EPOCHS):
        m.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx]; yb=ys_t[idx]
            if has:
                tj=torch.randint(0,len(Xf_t),(len(idx),)); xb=torch.cat([xb,Xf_t[tj]]); yb=torch.cat([yb,yf_t[tj]])
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); loss=crit(m(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m
def train_coral(Xs,ys,Xt,seed):
    set_seed(seed); m=CoralNet(nf,HID).to(device); crit=nn.CrossEntropyLoss(weight=class_w(ys))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys); Xt_t=torch.tensor(Xt)
    for e in range(EPOCHS):
        m.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx].to(device); yb=ys_t[idx].to(device)
            lg,zs=m(xb); loss=crit(lg,yb)
            if len(idx)>2:
                tj=torch.randint(0,len(Xt_t),(len(idx),)); _,zt=m(Xt_t[tj].to(device)); loss=loss+LAM_CORAL*coral_loss(zs,zt)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m
def train_dann(Xs,ys,Xt,seed):
    set_seed(seed); m=DANN(nf,HID).to(device); crit=nn.CrossEntropyLoss(weight=class_w(ys)); dcrit=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    Xs_t=torch.tensor(Xs); ys_t=torch.tensor(ys); Xt_t=torch.tensor(Xt)
    for e in range(EPOCHS):
        p=e/max(EPOCHS-1,1); lam=2.0/(1+np.exp(-10*p))-1; m.train(); pm=torch.randperm(len(Xs_t))
        for i in range(0,len(Xs_t),BS):
            idx=pm[i:i+BS]; xb=Xs_t[idx].to(device); yb=ys_t[idx].to(device)
            tj=torch.randint(0,len(Xt_t),(len(idx),)); xt=Xt_t[tj].to(device)
            ls,ds=m(xb,lam); _,dt=m(xt,lam)
            dl=torch.zeros(len(idx),dtype=torch.long,device=device); tl=torch.ones(len(idx),dtype=torch.long,device=device)
            loss=crit(ls,yb)+dcrit(ds,dl)+dcrit(dt,tl)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m
def pred(m,X,dann=False):
    m.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),16384):
            xb=torch.tensor(X[i:i+16384]).to(device)
            lg=m(xb,0.0)[0] if dann else (m(xb)[0] if isinstance(m,CoralNet) else m(xb))
            o.append(torch.softmax(lg.float(),1)[:,1].cpu().numpy())
    return np.concatenate(o)
def pick(Xtr,ytr,k,seed):
    rng=np.random.RandomState(seed); idxs=[]
    for c in [0,1]:
        ci=np.where(ytr==c)[0]
        if len(ci): idxs.append(rng.choice(ci,min(k,len(ci)),replace=False))
    idx=np.concatenate(idxs) if idxs else np.array([],dtype=int); return Xtr[idx],ytr[idx]
def ms(v): a=np.array(v,dtype=float); return np.nanmean(a),np.nanstd(a)

# ============ PART A: baseline vs CORAL vs DANN (5 seeds, mean±std) ============
print("\n"+"="*92)
print(f"PART A — BASELINE vs CORAL vs DANN | {N_SEEDS} seeds, AUROC mean±std (target base-rate)")
print("="*92)
for s,t in PAIRS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xs_=to_f32(sc.fit_transform(SP[s]['Xtr'])); ys=SP[s]['ytr']
    Xt_tr=to_f32(sc.transform(SP[t]['Xtr'])); Xt_te=to_f32(sc.transform(SP[t]['Xte'])); yt=SP[t]['yte']
    b,c,d=[],[],[]
    for sd in range(N_SEEDS):
        mb=train_mlp(Xs_,ys,np.empty((0,nf),np.float32),np.empty(0,np.int64),SEED+sd); b.append(AUROC(yt,pred(mb,Xt_te))); del mb
        mc=train_coral(Xs_,ys,Xt_tr,SEED+sd); c.append(AUROC(yt,pred(mc,Xt_te))); del mc
        md=train_dann(Xs_,ys,Xt_tr,SEED+sd); d.append(AUROC(yt,pred(md,Xt_te,dann=True))); del md
        gc.collect(); torch.cuda.empty_cache()
    bm,bs=ms(b); cm,cs=ms(c); dm,ds_=ms(d)
    print(f"  {SHORT[s]:>4}->{SHORT[t]:<4}(base={SP[t]['base']:.3f}) | "
          f"base {bm:.3f}±{bs:.3f} | CORAL {cm:.3f}±{cs:.3f} (Δ{cm-bm:+.3f}) | DANN {dm:.3f}±{ds_:.3f} (Δ{dm-bm:+.3f})")

# ============ PART B: few-shot (5 seeds, mean±std) ============
print("\n"+"="*92)
print(f"PART B — FEW-SHOT | {N_SEEDS} seeds | each cell = AUROC(mean±std) / AUPRC(mean±std)")
print("="*92)
ceil={}
for t in DATASETS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xin=to_f32(sc.fit_transform(SP[t]['Xtr'])); Xte=to_f32(sc.transform(SP[t]['Xte']))
    ro,pr=[],[]
    for sd in range(N_SEEDS):
        m=train_mlp(Xin,SP[t]['ytr'],np.empty((0,nf),np.float32),np.empty(0,np.int64),SEED+sd)
        p=pred(m,Xte); ro.append(AUROC(SP[t]['yte'],p)); pr.append(AUPRC(SP[t]['yte'],p)); del m; gc.collect()
    ceil[t]=(ms(ro),ms(pr))
for s,t in PAIRS:
    sc=QuantileTransformer(output_distribution='uniform',n_quantiles=1000,random_state=SEED)
    Xs_=to_f32(sc.fit_transform(SP[s]['Xtr'])); ys=SP[s]['ytr']
    Xt_tr=to_f32(sc.transform(SP[t]['Xtr'])); Xt_te=to_f32(sc.transform(SP[t]['Xte'])); yt=SP[t]['yte']
    print(f"  {SHORT[s]:>4}->{SHORT[t]:<4} (base={SP[t]['base']:.3f}):")
    for k in K_LIST:
        ro,pr=[],[]
        for sd in range(N_SEEDS):
            if k==0: Xf,yf=np.empty((0,nf),np.float32),np.empty(0,np.int64)
            else: Xf,yf=pick(Xt_tr,SP[t]['ytr'],k,SEED+sd)
            m=train_mlp(Xs_,ys,Xf,yf,SEED+sd); p=pred(m,Xt_te); ro.append(AUROC(yt,p)); pr.append(AUPRC(yt,p)); del m
            gc.collect(); torch.cuda.empty_cache()
        rm,rs=ms(ro); pm,ps=ms(pr)
        print(f"      k={k:<4} AUROC {rm:.3f}±{rs:.3f} | AUPRC {pm:.3f}±{ps:.3f}")
    (crm,crs),(cpm,cps)=ceil[t]
    print(f"      ceiling  AUROC {crm:.3f}±{crs:.3f} | AUPRC {cpm:.3f}±{cps:.3f}")

print("\nDONE. (Part A updates the CORAL/DANN numbers; Part B updates Table 6 — both with ±std over 5 seeds.)")
