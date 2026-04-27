import wandb
import json

api = wandb.Api()
runs = api.runs('SICAPv2_CONCH_maskLUT')

results = []
for r in runs[:5]:
    results.append({
        'name': r.name,
        'config': r.config,
        'summary': r.summary_metrics
    })

with open('get_runs.json', 'w') as f:
    json.dump(results, f, indent=2)
