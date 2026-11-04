"""Append-only SQLite episode/optimizer journals, separate from model weights."""
from dataclasses import asdict,replace
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
from ai.analysis_stats import Dataset, EpisodeRecord

SCHEMA=1
FIELDS=tuple(EpisodeRecord.__dataclass_fields__)[:11]


class HistoryStore:
    def __init__(self,root):
        self.root=Path(root)
        self.path=self.root/'history.sqlite3'
        self._connection=None
        self._optimization=[]

    def _writer(self):
        if self._connection is not None:
            return self._connection
        self.root.mkdir(parents=True,exist_ok=True)
        connection=sqlite3.connect(self.path,timeout=5)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,name TEXT,architecture TEXT,configuration TEXT,status TEXT);
            CREATE TABLE IF NOT EXISTS episodes(run_id TEXT,episode INTEGER,score REAL,length INTEGER,reward REAL,
              reason TEXT,cumulative_steps INTEGER,epsilon REAL,elapsed REAL,apples INTEGER,seed INTEGER,
              PRIMARY KEY(run_id,episode));
            CREATE TABLE IF NOT EXISTS optimization(run_id TEXT,step INTEGER,loss REAL,gradient_norm REAL,epsilon REAL,
              PRIMARY KEY(run_id,step));
        ''')
        connection.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('schema_version',str(SCHEMA)))
        if connection.execute('SELECT value FROM metadata WHERE key=?',('schema_version',)).fetchone()[0]!=str(SCHEMA):
            connection.close()
            raise ValueError('Unsupported history schema')
        self._connection=connection
        return connection

    def register(self,run,name,architecture,configuration):
        with self._writer() as db:
            if self._optimization:
                db.executemany('INSERT OR IGNORE INTO optimization VALUES (?,?,?,?,?)',self._optimization)
                self._optimization.clear()
            db.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?)',
                (run,name,json.dumps(architecture),json.dumps(configuration,allow_nan=False),'Running'))

    def append_episode(self,record):
        if type(record.episode) is not int or record.episode<1 or not math.isfinite(record.score):
            raise ValueError('Invalid episode record')
        values=tuple(asdict(record)[key] for key in FIELDS)
        with self._writer() as db:
            existing=db.execute('SELECT * FROM episodes WHERE run_id=? AND episode=?',(record.run_id,record.episode)).fetchone()
            if existing is not None and existing!=values:
                raise ValueError('History conflict: resuming an earlier checkpoint requires a new history branch')
            db.execute('INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)',values)
            if self._optimization:
                db.executemany('INSERT OR IGNORE INTO optimization VALUES (?,?,?,?,?)',self._optimization)
        self._optimization.clear()

    def append_optimization(self,run,step,loss,norm,epsilon):
        self._optimization.append((run,step,loss,norm,epsilon))
        if len(self._optimization)>=256:
            self.flush()

    def flush(self):
        if self._optimization:
            with self._writer() as db:
                db.executemany('INSERT OR IGNORE INTO optimization VALUES (?,?,?,?,?)',self._optimization)
            self._optimization.clear()

    def close(self):
        try:
            self.flush()
        finally:
            if self._connection is not None:
                self._connection.close()
                self._connection=None

    def status(self,run,status):
        with self._writer() as db:
            db.execute('UPDATE runs SET status=? WHERE id=?',(status,run))

    def datasets(self):
        if not self.path.is_file():
            return ()
        # mode=ro: discovery never creates, repairs or migrates a database.
        with closing(sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT id,name,architecture,configuration,status FROM runs ORDER BY rowid').fetchall()
        return tuple(dict(identifier=r[0],name=r[1],architecture=tuple(json.loads(r[2])),configuration=json.loads(r[3]),status=r[4]) for r in rows)

    def load(self,run):
        metadata=next((r for r in self.datasets() if r['identifier']==run),None)
        if metadata is None:
            raise ValueError('Training history not found')
        with closing(sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            records=tuple(EpisodeRecord(*row) for row in db.execute('SELECT * FROM episodes WHERE run_id=? ORDER BY episode',(run,)))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='episode_environments'").fetchone():
                environments={row[0]:row[1:] for row in db.execute('SELECT episode,environment_id,local_sequence FROM episode_environments WHERE run_id=?',(run,))}
                records=tuple(replace(r,environment_id=environments[r.episode][0],local_sequence=environments[r.episode][1]) if r.episode in environments else r for r in records)
        return Dataset(metadata['identifier'],metadata['name'],records,architecture=metadata['architecture'],
                       status=metadata['status'],configuration=metadata['configuration'],model_identity=metadata['configuration'].get('model_identity'))


def legacy_dataset(folder):
    """Read an immutable v1.4 checkpoint without constructing/loading a network."""
    import hashlib
    import torch
    import zipfile
    folder=Path(folder)
    meta_path=folder/'training.json'
    if folder.is_symlink() or meta_path.is_symlink() or meta_path.stat().st_size>200000:
        raise ValueError('Legacy metadata too large')
    metadata=json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(metadata,dict) or metadata.get('format_version')!=1 or metadata.get('kind')!='resumable-boundary':
        raise ValueError('Unsupported legacy checkpoint format')
    path=folder/'training.pt'
    if path.is_symlink() or path.stat().st_size>256000000:
        raise ValueError('Legacy state too large or not local')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('Legacy checkpoint checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>512000000:
            raise ValueError('Legacy checkpoint archive too large')
    state=torch.load(path,map_location='cpu',weights_only=True)
    scores=state['history']['scores']
    lengths=state['history'].get('lengths',[])
    rewards=state['history'].get('rewards',[])
    completed=metadata['episode']
    if type(completed) is not int or not isinstance(scores,list) or len(scores)>20000 or completed<len(scores):
        raise ValueError('Invalid legacy score history')
    for values in (lengths,rewards):
        if not isinstance(values,list) or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid legacy length/reward history')
    first=completed-len(scores)+1
    records=[]
    for index,score in enumerate(scores):
        if type(score) not in (int,float) or not math.isfinite(score):
            raise ValueError('Invalid legacy score')
        episode=first+index
        records.append(EpisodeRecord(metadata['model_identifier'],episode,score,
            lengths[index] if len(lengths)==len(scores) else None,
            rewards[index] if len(rewards)==len(scores) else None,
            state.get('last_reason') if episode==completed else None,
            metadata['environment_steps'] if episode==completed else None,
            metadata['epsilon'] if episode==completed else None,
            state.get('elapsed') if episode==completed else None,
            int(score)))  # Classic score records actual apple count; no hazard/star points.
    arch=metadata['architecture']
    return Dataset(metadata['model_identifier']+'/'+folder.name,metadata['model_name'],tuple(records),
        architecture=(arch['input_size'],*arch['hidden_sizes'],arch['output_size']),checkpoint=str(folder.resolve()),
        limitations=(f'v1.4 checkpoint: available episodes {first}..{completed}; no fabricated earlier history.',
            'Per-episode reasons, cumulative steps, epsilon and times were not recorded; only final values are available.',
            'Legacy raw histories share a tensor file with replay/optimizer data; read-only weights_only parsing is required.'),
        configuration=metadata['hyperparameters'],checkpoint_episode=completed)
