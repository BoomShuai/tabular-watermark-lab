"""Generate explicitly synthetic fixtures for an offline smoke run."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate(directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    n = 504
    pd.DataFrame({'dimension_0':rng.normal(0,20,n),'dimension_1':rng.normal(0,20,n),'target':np.arange(n)%2}).to_csv(directory/'synthetic_data.csv',index=False)
    pd.DataFrame({'Elevation':np.arange(n),'Aspect':np.arange(n)%360,'Slope':np.arange(n)%40,'feature':rng.normal(size=n),'Cover_Type':np.arange(n)%7+1}).to_csv(directory/'cover_type_with_columns.csv',index=False)
    pd.DataFrame({'0':rng.normal(size=n),'1':rng.normal(size=n),'target':np.arange(n)%10}).to_csv(directory/'digits_HOG.csv',index=False)
    pd.DataFrame({'RM':rng.normal(6,1,n),'LSTAT':rng.normal(12,4,n),'MEDV':rng.normal(25,8,n)}).to_csv(directory/'HousingData.csv',index=False)
    pd.DataFrame({'feature_a':np.arange(n)%2,'feature_b':np.arange(n)%4,'class_type':np.arange(n)%7+1}).to_csv(directory/'zoo.csv',index=False)
    (directory/'SYNTHETIC.json').write_text(json.dumps({'synthetic':True,'seed':seed,'rows':n,'scope':'Toy fixtures only; not the original public datasets or paper replication evidence.'},indent=2)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args(argv)
    try:
        generate(args.output,args.seed)
        print(json.dumps({'synthetic':True,'created':True}));return 0
    except (OSError,ValueError):
        print('Demo generation failed: choose a new output directory.');return 2


if __name__=='__main__':raise SystemExit(main())
