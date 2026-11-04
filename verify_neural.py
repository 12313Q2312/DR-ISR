"""Replay archived neural weights and validate splits, budgets and metric provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from neural.models import build_model
from neural.experiment import load_split


def verify(folder,data_dir="neural_data"):
    folder=Path(folder)
    cfg=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))["config"]
    torch.set_num_threads(cfg["threads"])
    rows=[json.loads(s) for s in (folder/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
    states=[json.loads(s) for s in (folder/"models.jsonl").read_text(encoding="utf-8").splitlines()]
    key=lambda r:(r["dataset"],r["group"],r["seed"],r["method"])
    models={key(r):r["state"] for r in states}
    expected={(d,g,s,m) for d in cfg["datasets"] for g in cfg["groups"] for s in cfg["seeds"] for m in cfg["methods"]}
    assert len(rows)==len(set(key(r) for r in rows)) and set(key(r) for r in rows)==expected
    assert len(states)==len(models) and set(models)==expected
    initial={}
    replayed=0
    max_prediction_absolute_error=0.0
    cached={}
    for row in rows:
        data_key=(row["dataset"],row["group"])
        if data_key not in cached:
            split,meta=load_split(data_dir,*data_key,cfg)
            saved=json.loads((folder/"splits"/("_".join(data_key)+".json")).read_text(encoding="utf-8"))
            assert meta==saved,"Split/scaler/source mismatch"
            assert len(set(meta["train_ids"])&set(meta["test_ids"]))==0
            assert len(set(meta["validation_ids"])&set(meta["test_ids"]))==0
            cached[data_key]=(split,meta)
        split,meta=cached[data_key]
        assert row["parameters"]<=cfg["parameter_cap"]
        assert row["optimizer_updates"]==cfg["training_steps"] or row["status"]!="ok"
        if row["search"]:
            s=row["search"]
            assert s["budget_used"]==cfg["search"]["budget"]
            assert s["budget_used"]==sum(s[k] for k in ("fitness_evaluations","semantic_probes","context_passes"))
            pair=data_key+(row["seed"],)
            assert initial.setdefault(pair,s["initial_population_sha256"])==s["initial_population_sha256"]
        torch.manual_seed(row["seed"])
        model,width,count=build_model(row["method"],split["train"][0].shape[1],row["degrees"],
                                      cfg["parameter_cap"],cfg["grid_size"],cfg["spline_order"])
        assert count==row["parameters"] and width==row["width"]
        template=model.state_dict()
        model.load_state_dict({k:torch.tensor(v,dtype=template[k].dtype) for k,v in models[key(row)].items()})
        model.eval()
        with torch.no_grad():
            for name,(x,y) in split.items():
                pred=model(torch.tensor(x,dtype=torch.float32)).numpy().reshape(-1)
                nmse=float(np.mean((pred-y)**2))
                assert np.isclose(nmse,row[name+"_nmse"],rtol=1e-5,atol=1e-7),(key(row),name,nmse)
                assert np.isclose(row[name+"_mse_raw"],nmse*meta["target_std"]**2,rtol=1e-5,atol=1e-7)
                if name=="test":
                    saved_prediction=np.asarray(row["test_predictions_normalized"])
                    max_prediction_absolute_error=max(max_prediction_absolute_error,float(np.max(np.abs(pred-saved_prediction))))
                    # PyTorch 1.13's float32 assert_close defaults use atol=1e-5.
                    # Windows/Linux BLAS reduction order can differ near zero.
                    # Keep the independent NMSE/raw-MSE checks above unchanged.
                    np.testing.assert_allclose(pred,saved_prediction,rtol=1e-5,atol=1e-5,err_msg=str(key(row)))
                replayed+=1
        assert row["best_validation_step"]==min(row["history"],key=lambda h:h["validation_nmse"])["step"]
    indexed={key(r):r for r in rows}
    for row in rows:
        if row["method"]!="tn_dr":continue
        other=indexed.get((row["dataset"],row["group"],row["seed"],"tn_vsr"))
        if other and row["degrees"]==other["degrees"]:
            assert row["initial_state_sha256"]==other["initial_state_sha256"],"Matched TN initialization differs"
            assert np.isclose(row["test_nmse"],other["test_nmse"],rtol=1e-10,atol=1e-12),"Matched TN training differs"
    provenance=json.loads((Path(data_dir)/"provenance.json").read_text(encoding="utf-8"))
    for path,digest in provenance["files"].items():
        assert hashlib.sha256((Path(data_dir)/path).read_bytes()).hexdigest()==digest,path
    checksums=json.loads((folder/"checksums.json").read_text(encoding="utf-8"))
    for path,digest in checksums.items():assert hashlib.sha256((folder/path).read_bytes()).hexdigest()==digest,path
    print(f"PASS: {len(rows)} models, {replayed} metric replays, splits/scalers/search budgets/parameter ceilings and {len(checksums)} file checksums; max prediction abs difference={max_prediction_absolute_error:.8g}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--data",default="neural_data")
    a=p.parse_args()
    verify(a.folder,a.data)
