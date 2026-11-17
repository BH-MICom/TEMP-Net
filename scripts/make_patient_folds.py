#!/usr/bin/env python3
"""Create seeded random patient-level five-fold assignments."""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=20261112); args=p.parse_args()
 with (args.data/'manifests/segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 patients=sorted({r['patient_group'] for r in rows}); rng=__import__('numpy').random.default_rng(args.seed); rng.shuffle(patients)
 assignments=[patients[i::args.folds].tolist() if hasattr(patients[i::args.folds],'tolist') else list(patients[i::args.folds]) for i in range(args.folds)]
 payload={'seed':args.seed,'n_folds':args.folds,'folds':{str(i):sorted(v) for i,v in enumerate(assignments)},'patient_to_fold':{p:i for i,v in enumerate(assignments) for p in v}}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
