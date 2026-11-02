"""Metadata-first discovery; journals and evaluations never load neural weights."""
from dataclasses import asdict,dataclass,replace
import json
from pathlib import Path
import sqlite3
from ai.analysis_stats import Dataset,EpisodeRecord
from ai.history import HistoryStore,legacy_dataset
from services.checkpoint_catalog import checkpoint_roots
from ai.model_registry import identity,parse_model_config


@dataclass(frozen=True)
class Choice:
    identifier: str
    name: str
    kind: str
    source: Path
    run: str
    checkpoint: Path | None = None
    model_id: str = ''
    model_name: str = ''
    episode: int | None = None

    @property
    def model_key(self):
        return self.model_id or self.run


class AnalysisRepository:
    def __init__(self,profile):
        self.profile=Path(profile)
        self.history=HistoryStore(self.profile/'history')

    def signature(self):
        """Cheap background invalidation for journals and newly published results."""
        paths=[self.history.path,Path(str(self.history.path)+'-wal')]
        for root in (*checkpoint_roots(self.profile/'checkpoints'),self.profile/'evaluations',self.profile/'frontier_index'):
            name=root.name
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
        for root in checkpoint_roots(self.profile/'checkpoints'):
            if not root.exists():continue
            for run in sorted(root.iterdir()):
                if not run.is_dir() or run.is_symlink():continue
                for folder in sorted(run.iterdir()):
                    path=folder/'training.json'
                    if not folder.is_dir() or folder.is_symlink() or not path.is_file():continue
                    try:
                        if path.stat().st_size>200000:continue
                        metadata=json.loads(path.read_text(encoding='utf-8'))
                        if metadata.get('format_version') not in (1,2) or metadata.get('kind') not in ('resumable-boundary','model-snapshot'):continue
                        if type(metadata['episode']) is not int or metadata['episode']<0:continue
                        if metadata['model_identifier']!=run.name:continue
                        checkpoints.append((metadata['episode'],Choice(str(folder.resolve()),
                            f"{metadata['model_name']} / ep {metadata['episode']}",'checkpoint',folder.resolve(),run.name,folder.resolve(),
                            run.name,metadata['model_name'],metadata['episode'])))
                    except (ValueError,OSError,KeyError,TypeError):continue
        for metadata in self.history.datasets():
            configuration=metadata['configuration']
            checkpoint_run=configuration.get('checkpoint_run_id',metadata['identifier'])
            matching=[(ep,c) for ep,c in checkpoints if c.run==checkpoint_run]
            checkpoint=max(matching,key=lambda x:x[0])[1].checkpoint if matching else None
            result.append(Choice(metadata['identifier'],metadata['name']+' / journal','journal',self.history.path,metadata['identifier'],
                checkpoint,checkpoint_run,metadata['name']))
        result.extend(choice for _,choice in sorted(checkpoints,key=lambda x:x[0],reverse=True))
        evaluations=self.profile/'evaluations'
        if evaluations.is_dir():
            for path in sorted(evaluations.glob('*.json'),reverse=True):
                try:
                    if path.stat().st_size>50000000:continue
                    data=json.loads(path.read_text(encoding='utf-8'))
                    for index,model in enumerate(data['models']):
                        checkpoint=Path(model['checkpoint']).resolve()
                        result.append(Choice(str(path.resolve())+f':{index}',model['name']+' / EVAL '+str(len(data['seeds'])),
                            'evaluation',path,str(index),checkpoint,model['run_id'],model['name'],model['training_episode']))
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
                checkpoint=str(choice.checkpoint) if choice.checkpoint else None,configuration=dataset.configuration,limitations=limits,model_identity=dataset.model_identity)
        if choice.kind=='checkpoint':
            metadata=json.loads((choice.source/'training.json').read_text(encoding='utf-8'))
            if metadata.get('experiment_id') and metadata.get('history_root'):
                journal=HistoryStore(Path(metadata['history_root'])).load(metadata['history_run'])
                config=parse_model_config(metadata)
                info=identity(config)
                if journal.architecture!=config.sizes or journal.model_identity and journal.model_identity['encoder_id']!=info['encoder_id']:
                    raise ValueError('History and checkpoint architecture/encoder do not match')
                records=tuple(r for r in journal.records if r.episode<=choice.episode)
                return replace(journal,identifier=choice.identifier,name=choice.model_name,
                    records=records,checkpoint=str(choice.checkpoint),checkpoint_episode=choice.episode,
                    status='Training history through checkpoint',model_identity=info,
                    limitations=('Frontier recorded training episodes through this checkpoint; not frozen evaluation data.',))
            dataset=legacy_dataset(choice.source)
            # Bound every history to this immutable snapshot. Enrich only with
            # actual journal records whose counters and overlapping values agree.
            for metadata in self.history.datasets():
                if metadata['configuration'].get('checkpoint_run_id',metadata['identifier'])!=choice.model_key:continue
                journal=self.history.load(metadata['identifier'])
                records=tuple(r for r in journal.records if r.episode<=choice.episode)
                if not records or not dataset.records or journal.architecture!=dataset.architecture:continue
                if records[-1].episode!=choice.episode or records[-1].cumulative_steps!=dataset.records[-1].cumulative_steps:continue
                by_episode={r.episode:r for r in records}
                overlaps=[r for r in dataset.records if r.episode in by_episode]
                if not overlaps or any((r.score,r.length,r.reward)!=(by_episode[r.episode].score,
                    by_episode[r.episode].length,by_episode[r.episode].reward) for r in overlaps):continue
                complete=len(records)==choice.episode and records[0].episode==1
                actual=records if complete else tuple(by_episode.get(r.episode,r) for r in dataset.records)
                configuration={**(journal.configuration or {}),'training':dataset.configuration}
                return replace(dataset,records=actual,status='Training history through checkpoint',configuration=configuration,
                    limitations=(f'Actual recorded training history through checkpoint episode {choice.episode}; not a frozen-policy evaluation.',))
            return replace(dataset,status='Training history through checkpoint')
        if choice.kind=='evaluation':
            data=json.loads(choice.source.read_text(encoding='utf-8'))
            model=data['models'][int(choice.run)]
            records=tuple(EpisodeRecord(**row) for row in model['records'])
            return Dataset(choice.identifier,model['name'],records,kind='evaluation',architecture=tuple(model['architecture']),
                status='Frozen greedy evaluation',checkpoint=model['checkpoint'],
                configuration={'evaluation':data['configuration'],'evaluation_seeds':data['seeds']},
                checkpoint_episode=model['training_episode'],model_identity=model.get('model_identity'),
                limitations=('Evaluation seeds/settings are recorded; these records are not exploratory training episodes.',))
        raise ValueError('Unknown analysis dataset')
