"""Render two measured acceptance examples from archived evidence."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dr_isr.tree import Node, evaluate

folder = Path("evidence/benchmark-v1")
rows = [json.loads(s) for s in (folder/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
methods = ["gp", "residual_gp", "shuffled_gp"]
hits = [sum(r["method"]==m and r["test_mse"]<1e-10 for r in rows) for m in methods]
bars = axes[0].bar(["GP", "Residual-guided", "Shuffled control"], hits,
                   color=["#2b6a9b", "#d77929", "#648542"], width=.6)
for bar, count in zip(bars,hits):
    axes[0].text(bar.get_x()+bar.get_width()/2,count+.8,f"{count}/50",ha="center")
axes[0].set_ylim(0,50)
axes[0].set_ylabel("Runs with test MSE < 1e-10")
axes[0].set_title("150 measured runs / equal charged budget")
x=np.linspace(-2,2,401)[:,None]
axes[1].axvspan(-1,1,color="#e5ebe7",label="Training-input range")
axes[1].plot(x[:,0],np.sin(x[:,0]**2),color="#233a35",lw=3,label="Target: sin(x squared)")
for method,style in [("gp","--"),("residual_gp",":")]:
    r=next(r for r in rows if r["task"]=="composition" and r["seed"]==2 and r["method"]==method)
    axes[1].plot(x[:,0],evaluate(Node.from_dict(r["tree"]),x),style,lw=2.3,
                 color="#2b6a9b" if method=="gp" else "#d77929",label="GP" if method=="gp" else "Residual: x*sin(x)")
axes[1].set_title("Composition / seed 2: a retained failure")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Output")
axes[1].legend(fontsize=8,loc="lower center")
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.15)
fig.tight_layout()
fig.savefig("docs/demo_overview.png",dpi=160)
fig.savefig("docs/demo_overview.svg")
p = Path("docs/demo_overview.svg")
p.write_bytes(("\n".join(line.rstrip() for line in p.read_text(encoding="utf-8").splitlines())+"\n").encode("utf-8"))
