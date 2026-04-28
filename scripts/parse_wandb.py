import os
import yaml
import json
from pathlib import Path

import sicap_imports  # noqa: F401
from sicap_imports import REPO_ROOT

wandb_dir = REPO_ROOT / "wandb"
runs = []
for p in wandb_dir.iterdir():
    if p.is_dir() and p.name.startswith("run-"):
        conf_path = p / "files" / "config.yaml"
        sum_path = p / "files" / "wandb-summary.json"
        
        if conf_path.exists() and sum_path.exists():
            with open(conf_path, 'r') as f:
                conf = yaml.safe_load(f)
            with open(sum_path, 'r') as f:
                summary = json.load(f)
            
            unfreeze = conf.get("unfreeze_last", {}).get("value", None)
            name = conf.get("_wandb", {}).get("value", {}).get("e", {}).get(list(conf.get("_wandb", {}).get("value", {}).get("e", {}).keys())[0], {}).get("args", [])
            # trying to get wandb-name if provided
            wname = ""
            if "--wandb-name" in name:
                idx = name.index("--wandb-name")
                wname = name[idx+1]
                
            runs.append({
                'run_id': p.name,
                'unfreeze_last': unfreeze,
                'wandb_name': wname,
                'agg_macro_f1': summary.get("aggregated/macro_f1", None),
                'Val1_macro_f1': summary.get("Val1/macro_f1", None),
                'Val2_macro_f1': summary.get("Val2/macro_f1", None),
                'Val3_macro_f1': summary.get("Val3/macro_f1", None),
                'Val4_macro_f1': summary.get("Val4/macro_f1", None),
                'agg_f1_GG3': summary.get("aggregated/f1_GG3", None),
                'agg_f1_GG4': summary.get("aggregated/f1_GG4", None),
                'agg_f1_GG5': summary.get("aggregated/f1_GG5", None),
                'train_loss_ratio': summary.get("Val1/val_train_loss_ratio", None)
            })

# Sort by name to get latest runs easily
runs = sorted(runs, key=lambda x: x['run_id'], reverse=True)

with open("runs_summary.json", 'w') as f:
    json.dump(runs[:10], f, indent=2)
