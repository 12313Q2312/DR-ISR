"""End-to-end controlled TN / residual TN / efficient-kan synthetic study."""
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
import numpy as np
import torch
from .models import build_model
from .vector_gp import SearchConfig,search


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def load_split(data_dir,dataset,group,config):
    path=Path(data_dir)/f"{dataset}_{group}.csv"
    data=np.loadtxt(path,delimiter=",",skiprows=1,ndmin=2)
    ids=np.random.default_rng(config["split_seed"]).permutation(len(data))
    n_train=int(len(data)*config["train_fraction"])
    n_val=int(len(data)*config["validation_fraction"])
    train_ids,val_ids,test_ids=np.split(ids,[n_train,n_train+n_val])
    x,y=data[:,:-1],data[:,-1]
    lo=x[train_ids].min(axis=0)
    span=x[train_ids].max(axis=0)-lo
    span=np.where(span<1e-12,1.,span)
    mean=float(y[train_ids].mean())
    std=max(float(y[train_ids].std()),1e-12)
    sx=2*(x-lo)/span-1
    sy=(y-mean)/std
    split={name:(sx[ind],sy[ind]) for name,ind in (("train",train_ids),("validation",val_ids),("test",test_ids))}
    metadata=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  train_ids=train_ids.tolist(),validation_ids=val_ids.tolist(),test_ids=test_ids.tolist(),
                  feature_min=lo.tolist(),feature_span=span.tolist(),target_mean=mean,target_std=std)
    return split,metadata


def state_to_json(model):return {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}


def train_model(model,split,config):
    tensor={k:(torch.tensor(x,dtype=torch.float32),torch.tensor(y[:,None],dtype=torch.float32)) for k,(x,y) in split.items() if k!="test"}
    x,y=tensor["train"]
    vx,vy=tensor["validation"]
    start=perf_counter()
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    best_loss=float("inf")
    best_state=None
    best_step=0
    history=[]
    updates=0
    failure=None
    def checkpoint(step):
        nonlocal best_loss,best_state,best_step
        model.eval()
        with torch.no_grad():
            train_mse=float(torch.mean((model(x)-y)**2))
            val_mse=float(torch.mean((model(vx)-vy)**2))
        if np.isfinite(val_mse) and np.isfinite(train_mse):
            history.append(dict(step=step,train_nmse=train_mse,validation_nmse=val_mse))
            if val_mse<best_loss:
                best_loss,best_state,best_step=val_mse,copy.deepcopy(model.state_dict()),step
        model.train()
    checkpoint(0)
    for step in range(1,config["training_steps"]+1):
        optimizer.zero_grad()
        loss=torch.mean((model(x)-y)**2)
        if not torch.isfinite(loss):failure="nonfinite training loss";break
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            failure="nonfinite gradient";break
        optimizer.step()
        updates+=1
        if step%config["checkpoint_every"]==0 or step==config["training_steps"]:checkpoint(step)
    if best_state is None:raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    seconds=perf_counter()-start
    model.eval()
    metrics={}
    with torch.no_grad():
        for name,(ax,ay) in split.items():
            prediction=model(torch.tensor(ax,dtype=torch.float32)).numpy().reshape(-1)
            metrics[name+"_nmse"]=float(np.mean((prediction-ay)**2))
            if name=="test":metrics["test_predictions_normalized"]=prediction.tolist()
    return dict(**metrics,training_seconds=seconds,optimizer_updates=updates,
                best_validation_step=best_step,status="ok" if failure is None else "partial_failure",
                failure=failure,history=history)


def run(config_path,data_dir,output):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    (out/"splits").mkdir()
    def git(*args):
        r=subprocess.run(["git",*args],capture_output=True,text=True,check=False)
        return r.stdout.strip()
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),config=cfg,
                  python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
                  device="cpu",git_commit=git("rev-parse","HEAD"),git_dirty=git("status","--porcelain"),
                  sources={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [*Path("neural").rglob("*.py"),Path("run_neural.py")]},
                  timing="Search + model construction + training (including validation checkpoints); excludes shared data preparation and final holdout inference")
    write_json(out/"manifest.json",manifest)
    rows=[]
    for dataset in cfg["datasets"]:
        for group in cfg["groups"]:
            split,metadata=load_split(data_dir,dataset,group,cfg)
            key=dataset+"_"+group
            write_json(out/"splits"/(key+".json"),metadata)
            for seed in cfg["seeds"]:
                order=list(cfg["methods"])
                random.Random(f"{key}/{seed}/neural-v1").shuffle(order)
                initial_population=None
                for method in order:
                    search_record=None
                    if method in ("tn_vsr","tn_dr"):
                        search_record=search(*split["train"],method,seed,SearchConfig(**cfg["search"]))
                        degrees=search_record["degrees"]
                        assert initial_population in (None,search_record["initial_population_sha256"])
                        initial_population=search_record["initial_population_sha256"]
                    else:degrees=[]
                    # Common model seed; same discovered powers => identical TN initialization.
                    torch.manual_seed(seed)
                    begin=perf_counter()
                    model,width,parameters=build_model(method,split["train"][0].shape[1],degrees,
                        cfg["parameter_cap"],cfg["grid_size"],cfg["spline_order"])
                    construction_seconds=perf_counter()-begin
                    initial_state=state_to_json(model)
                    result=train_model(model,split,cfg)
                    row=dict(dataset=dataset,group=group,seed=seed,method=method,run_order=order.index(method),
                             degrees=degrees,width=width,parameters=parameters,search=search_record,
                             model_construction_seconds=construction_seconds,
                             initial_state_sha256=hashlib.sha256(json.dumps(initial_state,sort_keys=True).encode()).hexdigest(),**result)
                    for part in ("train","validation","test"):
                        row[part+"_mse_raw"]=row[part+"_nmse"]*metadata["target_std"]**2
                    row["total_seconds"]=row["training_seconds"]+construction_seconds+(search_record["search_seconds"] if search_record else 0)
                    state=dict(dataset=dataset,group=group,seed=seed,method=method,state=state_to_json(model))
                    with (out/"models.jsonl").open("a",encoding="utf-8",newline="\n") as handle:
                        handle.write(json.dumps(state,allow_nan=False,separators=(",",":"))+"\n")
                    with (out/"runs.jsonl").open("a",encoding="utf-8",newline="\n") as handle:
                        handle.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                    rows.append(row)
                    print(f"{dataset}/{group} seed={seed} {method:6} test NMSE={row['test_nmse']:.5g} "
                          f"p={parameters} powers={degrees} total={row['total_seconds']:.2f}s {row['status']}",flush=True)
    manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
    write_json(out/"manifest.json",manifest)
    summarize(rows,out)
    checksums={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out/"checksums.json",checksums)
    return rows


def summarize(rows,out):
    methods=list(dict.fromkeys(r["method"] for r in rows))
    groups=[]
    paired=[]
    lines=["# Measured neural comparison", "", "Controlled synthetic experiment; not a reproduction of the paper's 16 real-world datasets.","",
           "TN-VSR and TN-DR use the paper's polynomial-neuron parameterization with a reimplemented, bounded homogeneous GP search. "
           "KAN is the attributed efficient-kan B-spline implementation, not the full pykan pipeline. "
           "Shared parameter CEILING and optimizer updates do not imply identical capacity or compute.","",
           "NMSE = MSE / variance of training labels. Data scaling, structure search and model training use training data only; "
           "validation selects checkpoints, test is used only for final evaluation. Labels remain noisy. No extrapolation experiment here.","",
           "| Dataset/group | Method | Seeds | Test NMSE median [Q1,Q3] | Parameters median | Search seconds median | Total seconds median |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for dataset,group in dict.fromkeys((r["dataset"],r["group"]) for r in rows):
        task_rows=[r for r in rows if (r["dataset"],r["group"])==(dataset,group)]
        for method in methods:
            subset=[r for r in task_rows if r["method"]==method]
            if not subset:continue
            values=np.array([r["test_nmse"] for r in subset])
            q=np.quantile(values,[.25,.5,.75])
            item=dict(dataset=dataset,group=group,method=method,n=len(subset),test_nmse_q1=float(q[0]),
                      test_nmse_median=float(q[1]),test_nmse_q3=float(q[2]),
                      parameters_median=float(np.median([r["parameters"] for r in subset])),
                      search_seconds_median=float(np.median([r["search"]["search_seconds"] if r["search"] else 0 for r in subset])),
                      total_seconds_median=float(np.median([r["total_seconds"] for r in subset])))
            groups.append(item)
            lines.append(f"| {dataset}/{group} | {method} | {len(subset)} | {q[1]:.6g} [{q[0]:.4g},{q[2]:.4g}] | "
                         f"{item['parameters_median']:g} | {item['search_seconds_median']:.3f} | {item['total_seconds_median']:.3f} |")
        for left,right in (("tn_vsr","kan"),("tn_dr","kan"),("tn_dr","tn_vsr"),("tn_vsr","mlp")):
            a={r["seed"]:r for r in task_rows if r["method"]==left}
            b={r["seed"]:r for r in task_rows if r["method"]==right}
            delta=[a[s]["test_nmse"]-b[s]["test_nmse"] for s in sorted(set(a)&set(b))]
            if delta:paired.append(dict(dataset=dataset,group=group,left=left,right=right,n=len(delta),
                median_difference=float(np.median(delta)),wins=sum(d< -1e-8 for d in delta),
                ties=sum(abs(d)<=1e-8 for d in delta),losses=sum(d>1e-8 for d in delta)))
    lines += ["","## Paired test-NMSE differences","","Negative favors the left model. Absolute tie tolerance 1e-8. Descriptive only.","",
              "| Dataset/group | Comparison | Median difference | Wins / ties / losses |","|---|---|---:|---|"]
    for p in paired:lines.append(f"| {p['dataset']}/{p['group']} | {p['left']} - {p['right']} | {p['median_difference']:.6g} | {p['wins']}/{p['ties']}/{p['losses']} |")
    lines += ["","All seeds and failures are retained. A shared fixed optimizer schedule is a protocol, not proof that each model was optimally tuned. "
              "Search costs are extra costs paid only by TN models, included in total time; KAN is not charged fictitious search evaluations. "
              "A single data split per synthetic task limits generalization of these findings."]
    Path(out,"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    write_json(Path(out,"summary.json"),dict(groups=groups,paired=paired))
