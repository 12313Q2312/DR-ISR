"""Import synthetic author data without modifying the existing checkout/environment."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--source",default="G:/Task_based_neurons/data")
parser.add_argument("--output",default="neural_data")
args=parser.parse_args()
out=Path(args.output)
out.mkdir(exist_ok=False)
manifest={"source_repository":"https://github.com/NewT123-WM/Task_based_neurons",
          "source_files":{},"files":{}}
for name in ("epsilon_low","epsilon_high"):
    src=Path(args.source)/(name+".h5")
    manifest["source_files"][src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    with h5py.File(src,"r") as handle:
        for group in sorted(handle.keys()):
            x=np.asarray(handle[group]["x_train"],dtype=float)
            y=np.asarray(handle[group]["y_train"],dtype=float).reshape(-1,1)
            path=out/f"{name}_{group}.csv"
            with path.open("w",encoding="utf-8",newline="\n") as stream:
                np.savetxt(stream,np.column_stack([x,y]),delimiter=",",fmt="%.17g",
                           header=",".join([f"x{i}" for i in range(x.shape[1])]+["y"]),comments="")
            manifest["files"][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(out/"provenance.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
print(f"Imported {len(manifest['files'])} synthetic datasets into {out}")
