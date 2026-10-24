"""Keep Toyota and Nissan listings, plus any newly scraped Honda listings. Never fabricate listings."""
from pathlib import Path
import pandas as pd
DATA=Path(__file__).resolve().parent/'data'
source=DATA/'used_cars_cleaned.csv'
if not source.exists(): raise SystemExit(f'Missing {source}')
df=pd.read_csv(source)
allowed={'Toyota':{'Camry','Corolla','RAV4'},'Nissan':{'Altima','Sentra','Rogue'},'Honda':{'Civic','Accord','CR-V'}}
mask=df.apply(lambda r:str(r['model']).strip() in allowed.get(str(r['make']).strip(),set()),axis=1)
removed=df[~mask]; retained=df[mask]
backup=DATA/'used_cars_before_honda_switch.csv'
if not backup.exists(): df.to_csv(backup,index=False)
retained.to_csv(source,index=False)
print(f'Retained {len(retained)} listings; removed {len(removed)}; original backed up at {backup}')
if 'source' in retained: print(retained['source'].value_counts().to_string())
print('No Honda listings were scraped by this script. Collect Honda listings separately and merge/deduplicate before matching.')
