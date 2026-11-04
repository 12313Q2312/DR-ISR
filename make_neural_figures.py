"""Static scientific figures from the frozen end-to-end experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument("folder")
args=p.parse_args()
out=Path(args.folder)
rows=[json.loads(s) for s in (out/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
methods=["tn_vsr","tn_dr","kan","mlp"]
names={"tn_vsr":"TN-VSR (controlled)","tn_dr":"TN-DR","kan":"B-spline KAN","mlp":"MLP"}
colors={"tn_vsr":"#2b6a9b","tn_dr":"#d77929","kan":"#49805e","mlp":"#7d729e"}
datasets=list(dict.fromkeys(r["dataset"] for r in rows))
groups=list(dict.fromkeys(r["group"] for r in rows))
fig,axes=plt.subplots(1,len(datasets),figsize=(6.8*len(datasets),4),squeeze=False,sharey=True)
for ax,dataset in zip(axes[0],datasets):
    for offset,method in enumerate(methods):
        selected=[[r["test_nmse"] for r in rows if (r["dataset"],r["group"],r["method"])==(dataset,group,method)] for group in groups]
        q1,med,q3=np.quantile(selected,[.25,.5,.75],axis=1)
        xs=np.arange(len(groups))+(offset-1.5)*.14
        ax.errorbar(xs,med,yerr=[med-q1,q3-med],marker="o",linestyle="none",color=colors[method],capsize=2,markersize=4,label=names[method])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(groups)),groups,rotation=30)
    ax.set_title(dataset+" / five seeds per model")
    ax.set_xlabel("Author synthetic group")
    ax.grid(axis="y",alpha=.2)
axes[0,0].set_ylabel("Test normalized MSE (median, IQR)")
axes[0,-1].legend(fontsize=8,loc="lower left")
fig.tight_layout()
fig.savefig(out/"test_errors.png",dpi=160)
fig.savefig(out/"test_errors.svg")
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8))
xs=np.arange(len(methods))
bottom=np.zeros(len(methods))
for component,label,color in [("search","Structure search","#b9cbbd"),("model_construction_seconds","Model construction","#829788"),("training_seconds","Train + validation","#2f6150")]:
    vals=[]
    for method in methods:
        group=[r for r in rows if r["method"]==method]
        vals.append(np.mean([r["search"]["search_seconds"] if r["search"] else 0 for r in group]) if component=="search" else np.mean([r[component] for r in group]))
    axes[0].bar(xs,vals,bottom=bottom,color=color,label=label)
    bottom+=vals
axes[0].set_xticks(xs,["TN-VSR","TN-DR","KAN","MLP"])
axes[0].set_ylabel("Mean seconds per run")
axes[0].set_title("Measured end-to-end cost")
axes[0].legend(fontsize=8)
for method in methods:
    group=[r["parameters"] for r in rows if r["method"]==method]
    axes[1].scatter([methods.index(method)]*len(group),group,s=10,color=colors[method],alpha=.2)
    axes[1].scatter([methods.index(method)],[np.median(group)],marker="_",s=350,color=colors[method])
axes[1].axhline(256,color="#68746e",linestyle="--",lw=1,label="Shared ceiling: 256")
axes[1].set_xticks(xs,["TN-VSR","TN-DR","KAN","MLP"])
axes[1].set_ylim(0,285)
axes[1].set_ylabel("Actual trainable parameter count")
axes[1].set_title("Ceiling matched; actual counts differ")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.15)
fig.tight_layout()
fig.savefig(out/"cost_and_capacity.png",dpi=160)
fig.savefig(out/"cost_and_capacity.svg")
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,3.8))
for method in methods:
    group=[r for r in rows if r["method"]==method]
    common=set(h["step"] for h in group[0]["history"])
    for row in group:common &= {h["step"] for h in row["history"]}
    steps=sorted(common)
    values=[[next(h["validation_nmse"] for h in row["history"] if h["step"]==step) for step in steps] for row in group]
    q1,med,q3=np.quantile(values,[.25,.5,.75],axis=0)
    ax.plot(steps,med,color=colors[method],label=names[method])
    ax.fill_between(steps,q1,q3,color=colors[method],alpha=.12)
ax.set_yscale("log")
ax.set_xlabel("Full-batch optimizer updates")
ax.set_ylabel("Validation NMSE (pooled median, IQR)")
ax.set_title("Recorded learning curves / no convergence claim")
ax.legend(fontsize=8)
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(out/"validation_curves.png",dpi=160)
fig.savefig(out/"validation_curves.svg")
for path in out.glob("*.svg"):
    path.write_bytes(("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n").encode("utf-8"))
