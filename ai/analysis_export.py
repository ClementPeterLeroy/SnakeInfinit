"""Exclusive-create UTF-8 exports with selected-range and calculation metadata."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import re
import unicodedata
from ai.analysis_stats import groups,select

RAW_FIELDS=('model_id','model_name','dataset_type','run_id','episode','score','length','reward','reason',
            'cumulative_steps','epsilon','elapsed','apples','seed')
GROUP_FIELDS=('model_id','model_name','dataset_type','episode_start','episode_end','sample_count',
              'minimum','q1','median','mean','q3','maximum','stddev')


def export_filename(datasets,kind='raw',size=100,alignment='episode'):
    """Readable, bounded Windows-compatible names using actual checkpoint identity."""
    if not datasets or kind not in ('raw','grouped'):
        raise ValueError('Select data and a valid export type')
    def slug(text,limit=32):
        text=unicodedata.normalize('NFKC',str(text)).lower()
        return re.sub(r'[^\w-]+','-',text,flags=re.UNICODE).strip('-_')[:limit].rstrip('-_') or 'model'
    identities=[]
    for dataset in datasets:
        checkpoint=Path(dataset.checkpoint).name if dataset.checkpoint else None
        match=re.fullmatch(r'episode-(\d+)-([a-zA-Z0-9]+)',checkpoint or '')
        if match:
            tag=f'{int(match[1])}ep-{slug(match[2],8)}'
        elif dataset.checkpoint_episode is not None:
            tag=f'{dataset.checkpoint_episode}ep'
        else:
            tag=slug(checkpoint,24) if checkpoint else 'journal'
        identity=slug(dataset.name)+'_'+tag
        if dataset.kind=='evaluation':identity+='-eval'
        identities.append(identity)
    suffix='raw' if kind=='raw' else f'grouped-{size}'+('-steps' if alignment=='steps' else '')
    return 'snakeinfinite_'+'_vs_'.join(identities)+'_'+suffix+'.csv'


def export_named_csv(directory,datasets,kind='raw',start=None,end=None,size=100,alignment='episode'):
    """Exclusive creation also handles repeated requests and concurrent exports."""
    base=Path(directory)/export_filename(datasets,kind,size,alignment)
    number=1
    while True:
        path=base if number==1 else base.with_name(base.stem+f'-{number}'+base.suffix)
        try:
            return export_csv(path,datasets,kind,start,end,size,alignment)
        except FileExistsError:
            if not path.exists() and not path.with_suffix(path.suffix+'.metadata.json').exists():
                raise
            number+=1


def export_csv(path,datasets,kind='raw',start=None,end=None,size=100,alignment='episode'):
    if kind not in ('raw','grouped'):
        raise ValueError('Choose raw or grouped CSV')
    path=Path(path)
    sidecar=path.with_suffix(path.suffix+'.metadata.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Export exists; choose a new filename')
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema_version=1,datasets=[dict(id=d.identifier,name=d.name,kind=d.kind,limitations=d.limitations,
        checkpoint=d.checkpoint,checkpoint_episode=d.checkpoint_episode,model_identity=d.model_identity,
        architecture=d.architecture,parameter_count=d.parameter_count) for d in datasets],
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
