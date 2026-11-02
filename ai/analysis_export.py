"""Exclusive-create UTF-8 exports with selected-range and calculation metadata."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
from ai.analysis_stats import groups,select

RAW_FIELDS=('model_id','model_name','dataset_type','run_id','episode','score','length','reward','reason',
            'cumulative_steps','epsilon','elapsed','apples','seed')
GROUP_FIELDS=('model_id','model_name','dataset_type','episode_start','episode_end','sample_count',
              'minimum','q1','median','mean','q3','maximum','stddev')


def export_csv(path,datasets,kind='raw',start=None,end=None,size=100,alignment='episode'):
    if kind not in ('raw','grouped'):
        raise ValueError('Choose raw or grouped CSV')
    path=Path(path)
    sidecar=path.with_suffix(path.suffix+'.metadata.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Export exists; choose a new filename')
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema_version=1,datasets=[dict(id=d.identifier,name=d.name,kind=d.kind,limitations=d.limitations) for d in datasets],
        selected_range=[start,end],range_unit=alignment,grouping=size,alignment=alignment,
        quartiles='linear/type 7: h=(n-1)*p',standard_deviation='population, ddof=0',whiskers='true minimum/maximum')
    created=False
    try:
        with path.open('x',newline='',encoding='utf-8') as stream:
            created=True
            writer=csv.DictWriter(stream,fieldnames=RAW_FIELDS if kind=='raw' else GROUP_FIELDS)
            writer.writeheader()
            for dataset in datasets:
                records=select(dataset.records,start,end,alignment)
                rows=(asdict(r) for r in records) if kind=='raw' else groups(records,size,alignment)
                for row in rows:
                    payload=dict(model_id=dataset.identifier,model_name=dataset.name,dataset_type=dataset.kind,
                                 **{k:v for k,v in row.items() if k in writer.fieldnames})
                    writer.writerow({key:format(value,'.12g') if isinstance(value,float) else value for key,value in payload.items()})
        with sidecar.open('x',encoding='utf-8') as stream:
            json.dump(metadata,stream,indent=2,ensure_ascii=False,allow_nan=False)
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path
