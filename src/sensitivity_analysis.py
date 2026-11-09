#!/usr/bin/env python3
"""Camera-ready analyses: support-threshold sensitivity (Table IV), KEGG vs Reactome across settings, and per-split GPT-4.1 stability. Re-scores stored outputs; no API calls."""
import json, sys, itertools
from collections import defaultdict
import pathcheck as pc
R = str(pc.OUT_DIR) + "/"
kegg=pc.parse_enrichr(pc.DATA_DIR/"KEGG_2021_Human.txt","KEGG")
rea=pc.parse_enrichr(pc.DATA_DIR/"Reactome_2022.txt","Reactome")
lib={**kegg,**rea}
print("library sets",len(lib), "KEGG",len(kegg),"Reactome",len(rea))
universe=sorted({g for r in lib.values() for g in r["genes"]}); N=len(universe); print("N",N)
cases=json.load(open(R+"benchmark.json")); cmap={c["id"]:c for c in cases}
gpt=json.load(open(R+"openai_gpt-4.1_outputs.json"))
def preds_from_eval(f):
    ev=json.load(open(R+f)); return {e["id"]:[{"name":it["predicted_name"],"reason":it.get("reason")} for it in e["items"]] for e in ev}
gens={"GPT-4.1":{k:v for k,v in gpt.items()},"Random":preds_from_eval("eval_random.json"),"Closed-book":preds_from_eval("eval_closed_book.json"),"ORA":preds_from_eval("eval_ora.json")}
# cache matches
mcache={}
def best(name):
    if name not in mcache:
        h=pc.match_pathway(name,lib,topk=8); mcache[name]=h[0] if h else None
    return mcache[name]
def score(gen, cut=0.62, mo=3, mp=0.05, mf=0.08, exact=False, splits=None):
    ex=[];sp=[]
    for cid,preds in gens[gen].items():
        c=cmap[cid]
        if splits and c["split"] not in splits: continue
        if not preds: continue
        e=s=0
        for p in preds:
            name=p["name"] if isinstance(p,dict) else str(p)
            b=best(name)
            ok = b is not None and ((b["exact"]) if exact else b["score"]>=cut)
            if ok:
                e+=1
                q=set(x.upper() for x in c["genes"]); P=set(x.upper() for x in lib[b["key"]]["genes"])
                k=len(q&P); pv=pc.hypergeom_sf(k,len(P),len(q),N)
                sup = k>=mo and ((pv<=mp) if mp is not None else False or False)
                sup = k>=mo and ((mp is not None and pv<=mp) or (mf is not None and k/len(q)>=mf))
                s+=sup
        ex.append(e/len(preds)); sp.append(s/len(preds))
    return sum(ex)/len(ex), sum(sp)/len(sp), len(ex)

# ---- threshold sensitivity
import statistics
print("=== single-parameter variation (others default) : exist/support for GPT-4.1 | Random sup | Closed sup | ORA sup")
def row(lbl,**kw):
    r={g:score(g,**kw) for g in gens}
    print(f"{lbl:28} GPT ex={r['GPT-4.1'][0]:.3f} sup={r['GPT-4.1'][1]:.3f} gap={r['GPT-4.1'][0]-r['GPT-4.1'][1]:.3f} | Rand sup={r['Random'][1]:.3f} | CB ex={r['Closed-book'][0]:.3f} sup={r['Closed-book'][1]:.3f} | ORA sup={r['ORA'][1]:.3f}")
    return r
row("default")
for mo in [2,4,5]: row(f"min_overlap={mo}",mo=mo)
for mp in [0.01,0.001]: row(f"p<={mp}",mp=mp)
row("p only (no frac)",mf=None)
row("frac only (no p)",mp=None)
for mf in [0.05,0.12,0.15]: row(f"frac>={mf}",mf=mf)
for cut in [0.55,0.70,0.80]: row(f"match cut={cut}",cut=cut)
row("exact match",exact=True)
print("=== full grid GPT-4.1 support range")
res=[]
for mo in [2,3,4,5]:
  for mp in [0.01,0.05,0.10]:
    for mf in [None,0.05,0.08,0.12]:
      e,s,_=score("GPT-4.1",mo=mo,mp=mp,mf=mf); er,sr,_=score("Random",mo=mo,mp=mp,mf=mf); eo,so,_=score("ORA",mo=mo,mp=mp,mf=mf)
      res.append((mo,mp,mf,s,sr,so))
ss=[r[3] for r in res]; print("n settings",len(res),"GPT sup min %.3f max %.3f"%(min(ss),max(ss)), "random sup max %.3f"%max(r[4] for r in res), "ORA sup min %.3f"%min(r[5] for r in res))
print("GPT exist (fixed):", "%.3f"%score("GPT-4.1")[0])
lo=min(res,key=lambda r:r[3]); hi=max(res,key=lambda r:r[3]); print("lowest",lo,"highest",hi)
# split-level check of key qualitative findings under grid: noisy< clean? KEGG vs Reactome gap?
def by_source(gen,src,**kw):
    ids={c["id"] for c in cases if c["source"]==src}
    saved=gens[gen]; gens[gen]={k:v for k,v in saved.items() if k in ids}
    r=score(gen,**kw); gens[gen]=saved; return r
print("=== KEGG vs Reactome support under settings")
for kw in [{}, {"mo":2},{"mo":4},{"mp":0.01},{"mf":None},{"cut":0.70},{"exact":True}]:
    k=by_source("GPT-4.1","KEGG",**kw); r=by_source("GPT-4.1","Reactome",**kw)
    print(kw, "KEGG sup %.3f  Reactome sup %.3f"%(k[1],r[1]))

# ---- KEGG vs Reactome across settings
def by_source(gen,src,**kw):
    ids={c["id"] for c in cases if c["source"]==src}
    saved=gens[gen]; gens[gen]={k:v for k,v in saved.items() if k in ids}
    r=score(gen,**kw); gens[gen]=saved; return r
bad=0; n=0; mind=1
for mo in [2,3,4,5]:
  for mp in [0.01,0.05,0.10]:
    for mf in [None,0.05,0.08,0.12]:
      k=by_source("GPT-4.1","KEGG",mo=mo,mp=mp,mf=mf)[1]; r=by_source("GPT-4.1","Reactome",mo=mo,mp=mp,mf=mf)[1]
      n+=1; bad+= k<=r; mind=min(mind,k-r)
for cut in [0.55,0.70,0.80]:
    k=by_source("GPT-4.1","KEGG",cut=cut)[1]; r=by_source("GPT-4.1","Reactome",cut=cut)[1]; n+=1; bad+=k<=r; mind=min(mind,k-r)
k=by_source("GPT-4.1","KEGG",exact=True)[1]; r=by_source("GPT-4.1","Reactome",exact=True)[1]; n+=1; bad+=k<=r; mind=min(mind,k-r)
print("settings",n,"KEGG<=Reactome count",bad,"min gap %.3f"%mind)

# ---- per-split stability
st=json.load(open(R+"gpt41_stability_outputs.json"))
def keys(preds):
    out=[]
    for p in preds:
        b=best(p["name"] if isinstance(p,dict) else str(p))
        if b and b["score"]>=0.62: out.append(b["key"])
    return out
def jac(a,b):
    A,B=set(a),set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A&B)/len(A|B)
by=defaultdict(lambda: defaultdict(list)); src=defaultdict(lambda: defaultdict(list))
ids=set()
for k,rec in st.items():
    cid=rec["id"]; ids.add(cid); c=cmap[cid]
    j=jac(keys(gpt[cid]),keys(rec["pathways"]))
    by[rec["mode"]][c["split"]].append(j); by[rec["mode"]]["ALL"].append(j); src[rec["mode"]][c["source"]].append(j)
from collections import Counter
print("picked by split",Counter(cmap[i]["split"] for i in ids), "by source", Counter(cmap[i]["source"] for i in ids))
for m in ["shuffle","drop10","drop20","add10"]:
    print(m, {s:(len(v),round(sum(v)/len(v),3)) for s,v in by[m].items()}, {s:round(sum(v)/len(v),3) for s,v in src[m].items()})
# fraction identical sets under shuffle
v=by["shuffle"]["ALL"]; print("shuffle identical frac %.3f"%(sum(x==1.0 for x in v)/len(v)))
