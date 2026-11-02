"""Copy the latest trained models into the repository for sharing through Git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.userdata import user_data_dir


def export_model(source,destination,*,training=None):
    """Copy a portable inference model, preserving original weights and metadata."""
    source,destination=Path(source),Path(destination)
    metadata_path=source/'network.json'
    if metadata_path.is_symlink() or metadata_path.stat().st_size>64000:
        raise ValueError('Invalid model metadata')
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    filename=metadata.get('weights_file','')
    if metadata.get('format_version')!=1 or not re.fullmatch(r'weights-[0-9a-f]{32}\.pt',filename):
        raise ValueError('Unsupported model metadata')
    weights=source/filename
    if weights.is_symlink() or weights.stat().st_size>8000000:
        raise ValueError('Invalid model weights')
    payload=weights.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=metadata.get('sha256'):
        raise ValueError('Model weights checksum mismatch')
    files={'network.json':metadata_path.read_bytes(),filename:payload}
    if training:
        card={key:training[key] for key in ('model_identifier','model_name','episode','environment_steps',
            'architecture','hyperparameters','created_utc')}
        card.update(format_version=1,purpose='Portable frozen inference model',
            source_checkpoint=source.name,resumable=False,
            note='Original weights and network metadata. Resume state remains in the local checkpoint library.')
        files['model_card.json']=(json.dumps(card,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    if destination.exists():
        if destination.is_symlink() or any(not (destination/name).is_file() or (destination/name).read_bytes()!=data for name,data in files.items()):
            raise FileExistsError('Different export already exists: '+str(destination))
        return destination
    destination.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        with (destination/name).open('xb') as stream:stream.write(data)
    return destination


def export_latest(profile,destination):
    """One latest committed checkpoint per run, plus saved laboratory models."""
    profile,destination=Path(profile),Path(destination)
    latest={}
    for path in sorted((profile/'checkpoints').glob('*/*/training.json')):
        if path.is_symlink() or path.stat().st_size>200000:continue
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('format_version')!=1 or data.get('kind')!='resumable-boundary':continue
        run=data['model_identifier']
        if not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',run):
            raise ValueError('Invalid model run identifier')
        if type(data['episode']) is not int or data['episode']<0:
            raise ValueError('Invalid checkpoint episode')
        key=(data['episode'],data['created_utc'])
        if run not in latest or key>latest[run][0]:latest[run]=(key,path.parent,data)
    exported=[]
    for run,(_,source,data) in sorted(latest.items()):
        name=re.sub(r'[^A-Za-z0-9_-]+','_',data['model_name']).strip('_')[:15] or 'model'
        name=f"{name}-ep{data['episode']}-{run[:8]}"
        exported.append(export_model(source,destination/name,training=data))
    for path in sorted((profile/'models').glob('*/network.json')):
        name='lab-'+re.sub(r'[^A-Za-z0-9_-]+','_',path.parent.name)[:36]
        exported.append(export_model(path.parent,destination/name))
    return exported


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=user_data_dir())
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    args=parser.parse_args()
    for path in export_latest(args.profile,args.output):print(path)


if __name__=='__main__':main()
