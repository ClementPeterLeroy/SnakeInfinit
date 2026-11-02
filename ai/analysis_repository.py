"""Metadata-first discovery; journals and evaluations never load neural weights."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import sqlite3
from ai.analysis_stats import Dataset,EpisodeRecord
from ai.history import HistoryStore,legacy_dataset


@dataclass(frozen=True)
class Choice:
    identifier: str
    name: str
    kind: str
    source: Path
    run: str
    checkpoint: Path | None = None


class AnalysisRepository:
    def __init__(self,profile):
        self.profile=Path(profile)
        self.history=HistoryStore(self.profile/'history')

    def signature(self):
        """Cheap background invalidation for journals and newly published results."""
        paths=[self.history.path,Path(str(self.history.path)+'-wal')]
        for name in ('checkpoints','evaluations'):
            root=self.profile/name
            if root.exists():
                paths.extend(root.glob('*/*.json') if name=='checkpoints' else root.glob('*.json'))
                if name=='checkpoints':paths.extend(root.glob('*/*/training.json'))
        result=[]
        for path in paths:
            try:
                stat=path.stat()
                result.append((str(path),stat.st_size,stat.st_mtime_ns))
            except FileNotFoundError:continue
        return tuple(sorted(result))

    def choices(self):
        result=[]
        checkpoints=[]
        root=self.profile/'checkpoints'
        if root.exists():
            for run in sorted(root.iterdir()):
                if not run.is_dir() or run.is_symlink():continue
                for folder in sorted(run.iterdir()):
                    path=folder/'training.json'
                    if not folder.is_dir() or folder.is_symlink() or not path.is_file():continue
                    try:
                        if path.stat().st_size>200000:continue
                        metadata=json.loads(path.read_text(encoding='utf-8'))
                        checkpoints.append((metadata['episode'],Choice(str(folder.resolve()),
                            f"{metadata['model_name']} / ep {metadata['episode']}",'checkpoint',folder,run.name,folder)))
                    except (ValueError,OSError,KeyError,TypeError):continue
        for metadata in self.history.datasets():
            configuration=metadata['configuration']
            checkpoint_run=configuration.get('checkpoint_run_id',metadata['identifier'])
            matching=[(ep,c) for ep,c in checkpoints if c.run==checkpoint_run]
            checkpoint=max(matching,key=lambda x:x[0])[1].source if matching else None
            result.append(Choice(metadata['identifier'],metadata['name']+' / journal','journal',self.history.path,metadata['identifier'],checkpoint))
        result.extend(choice for _,choice in sorted(checkpoints,key=lambda x:x[0],reverse=True))
        evaluations=self.profile/'evaluations'
        if evaluations.is_dir():
            for path in sorted(evaluations.glob('*.json'),reverse=True):
                try:
                    if path.stat().st_size>50000000:continue
                    data=json.loads(path.read_text(encoding='utf-8'))
                    for index,model in enumerate(data['models']):
                        result.append(Choice(str(path.resolve())+f':{index}',model['name']+' / EVAL '+str(len(data['seeds'])),
                            'evaluation',path,str(index),Path(model['checkpoint'])))
                except (OSError,ValueError,KeyError,TypeError):continue
        return tuple(result)

    def load(self,choice):
        if choice.kind=='journal':
            dataset=self.history.load(choice.run)
            records=dataset.records
            limits=()
            if records and records[0].episode!=1:
                limits=(f'Resumed session: recorded episodes start at {records[0].episode}; previous records were not migrated.',)
            return Dataset(dataset.identifier,dataset.name,records,architecture=dataset.architecture,status=dataset.status,
                checkpoint=str(choice.checkpoint) if choice.checkpoint else None,configuration=dataset.configuration,limitations=limits)
        if choice.kind=='checkpoint':
            dataset=legacy_dataset(choice.source)
            # A new journal is loaded separately; checkpoint snapshots retain their available boundary window.
            return dataset
        if choice.kind=='evaluation':
            data=json.loads(choice.source.read_text(encoding='utf-8'))
            model=data['models'][int(choice.run)]
            records=tuple(EpisodeRecord(**row) for row in model['records'])
            return Dataset(choice.identifier,model['name'],records,kind='evaluation',architecture=tuple(model['architecture']),
                status='Frozen greedy evaluation',checkpoint=model['checkpoint'],
                configuration={'evaluation':data['configuration'],'evaluation_seeds':data['seeds']},
                limitations=('Evaluation seeds/settings are recorded; these records are not exploratory training episodes.',))
        raise ValueError('Unknown analysis dataset')
