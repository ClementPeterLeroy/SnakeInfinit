"""Persistent metadata-only family/generation/checkpoint registry and reversible Trash."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4
from ai.model_registry import identity,parse_model_config
from services.checkpoint_catalog import checkpoint_roots

FAMILIES={'Alpha':'A','Distance':'B','CNN':'C','oCNN':'D'}


def utc():return datetime.now(timezone.utc).isoformat()


def metadata(path,limit=250000):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>limit:raise ValueError('Unsafe or oversized metadata')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('Metadata must be an object')
    return value


def family_of(data):
    config=parse_model_config(data)
    code=identity(config)['model_configuration_id']
    return next(name for name,value in FAMILIES.items() if value==code)


def alive(pid):
    if pid==os.getpid():return True
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5  # Unqueryable live owner: protect it.
        try:
            code=ctypes.c_ulong()
            kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code))
            return code.value==259
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


class ModelLibrary:
    def __init__(self,profile,repository=None):
        self.profile=Path(profile).resolve();self.repository=Path(repository).resolve() if repository else None
        self.root=self.profile/'model_library'
        if self.root.is_symlink() or getattr(self.root,'is_junction',lambda:False)():raise ValueError('Library metadata cannot use a linked directory')
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'library.sqlite3'
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS library_schema(version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO library_schema VALUES (1);
                CREATE TABLE IF NOT EXISTS sequences(family TEXT PRIMARY KEY,next_number INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS generations(id TEXT PRIMARY KEY,family TEXT NOT NULL,number INTEGER NOT NULL,
                  description TEXT,created TEXT,status TEXT,data TEXT,UNIQUE(family,number));
                CREATE TABLE IF NOT EXISTS aliases(origin TEXT PRIMARY KEY,generation_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,generation_id TEXT NOT NULL,origin TEXT UNIQUE,
                  path TEXT,kind TEXT,episode INTEGER,steps INTEGER,status TEXT,data TEXT);
                CREATE TABLE IF NOT EXISTS sessions(run TEXT PRIMARY KEY,generation_id TEXT,root TEXT,interface TEXT,status TEXT);
                CREATE TABLE IF NOT EXISTS references_data(id TEXT PRIMARY KEY,generation_id TEXT,checkpoint_id TEXT,path TEXT,kind TEXT);
                CREATE TABLE IF NOT EXISTS leases(token TEXT PRIMARY KEY,generation_id TEXT,checkpoint_id TEXT,pid INTEGER,owner TEXT,kind TEXT);
                CREATE TABLE IF NOT EXISTS trash(id TEXT PRIMARY KEY,kind TEXT,target TEXT,state TEXT,previous TEXT,plan TEXT);
                CREATE TABLE IF NOT EXISTS managed_roots(path TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS locations(path TEXT PRIMARY KEY,checkpoint_id TEXT NOT NULL);
            ''')
            if db.execute('SELECT version FROM library_schema').fetchall()!=[(1,)]:raise ValueError('Unsupported Model Library schema')
        self.allow_root(self.profile/'checkpoints');self.allow_root(self.profile/'models');self.allow_root(self.profile/'frontier')
        self.allow_root(self.profile/'evaluations');self.allow_root(self.profile/'experiments');self.allow_root(self.profile/'exports')
        if self.repository:
            self.allow_root(self.repository/'experiments/frontier');self.allow_root(self.repository/'models')
        self._recover_operations()
        self._expire_leases()

    @contextmanager
    def db(self,write=False):
        connection=sqlite3.connect(self.path,timeout=15)
        connection.execute('PRAGMA journal_mode=WAL');connection.execute('PRAGMA foreign_keys=ON')
        try:
            if write:connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:connection.rollback();raise
        finally:connection.close()

    def allow_root(self,path):
        with self.db(True) as db:db.execute('INSERT OR IGNORE INTO managed_roots VALUES (?)',(str(Path(path).resolve()),))

    def _new(self,db,family,origin,description='',created=None,data=None):
        if family not in FAMILIES:raise ValueError('Unknown model family')
        if origin:
            old=db.execute('SELECT generation_id FROM aliases WHERE origin=?',(origin,)).fetchone()
            if old:return old[0]
        db.execute('INSERT OR IGNORE INTO sequences VALUES (?,1)',(family,))
        number=db.execute('SELECT next_number FROM sequences WHERE family=?',(family,)).fetchone()[0]
        db.execute('UPDATE sequences SET next_number=next_number+1 WHERE family=?',(family,))
        identifier=uuid4().hex
        db.execute('INSERT INTO generations VALUES (?,?,?,?,?,?,?)',(identifier,family,number,description[:160],created or (data or {}).get('created_utc') or utc(),'Created',json.dumps(data or {})))
        if origin:db.execute('INSERT INTO aliases VALUES (?,?)',(origin,identifier))
        return identifier

    def create(self,family,description=''):
        with self.db(True) as db:return self._new(db,family,None,description)

    def get(self,identifier):
        with self.db() as db:row=db.execute('SELECT * FROM generations WHERE id=?',(identifier,)).fetchone()
        if not row:raise ValueError('Unknown generation identity')
        keys=('id','family','number','description','created','status','data')
        # Database row has seven columns; canonical name is never user-editable.
        item=dict(zip(keys,row));item['data']=json.loads(item['data'])
        item['name']=f"Snake DQN {item['family']} — Gen {item['number']}"
        item['display_name']=item['name']+(' | '+item['description'] if item['description'] else '')
        return item

    def generations(self,family=None,include_deleted=False):
        with self.db() as db:ids=[row[0] for row in db.execute('SELECT id FROM generations ORDER BY family,number')]
        return [g for identifier in ids if (g:=self.get(identifier)) and (family is None or g['family']==family) and (include_deleted or g['status']!='Deleted')]

    def describe(self,identifier,description):
        if not isinstance(description,str) or len(description)>160:raise ValueError('Description must contain at most 160 characters')
        with self.db(True) as db:db.execute('UPDATE generations SET description=? WHERE id=?',(description.strip(),identifier))

    def _expire_leases(self):
        with self.db(True) as db:
            for token,pid in db.execute('SELECT token,pid FROM leases').fetchall():
                if not alive(pid):db.execute('DELETE FROM leases WHERE token=?',(token,))
            db.execute("UPDATE generations SET status='Interrupted' WHERE status IN ('Training','Paused') AND id NOT IN (SELECT generation_id FROM leases WHERE kind='training')")

    def acquire(self,generation,owner,checkpoint=None,training=False):
        self._expire_leases()
        with self.db(True) as db:
            row=db.execute('SELECT status FROM generations WHERE id=?',(generation,)).fetchone()
            if not row or row[0] in ('Deleted','Deleting','Restoring'):raise ValueError('Generation is unavailable')
            if training and db.execute("SELECT 1 FROM leases WHERE generation_id=? AND kind='training'",(generation,)).fetchone():raise ValueError('Generation already has an active training session')
            if checkpoint:
                cp=db.execute('SELECT generation_id,status FROM checkpoints WHERE id=?',(checkpoint,)).fetchone()
                if not cp or cp[0]!=generation or cp[1] in ('Deleted','Deleting'):raise ValueError('Checkpoint is stale or deleted')
            token=uuid4().hex
            db.execute('INSERT INTO leases VALUES (?,?,?,?,?,?)',(token,generation,checkpoint,os.getpid(),owner,'training' if training else 'loaded'))
            return token

    def release(self,token):
        if token:
            with self.db(True) as db:db.execute('DELETE FROM leases WHERE token=?',(token,))

    def status(self,generation,status,owner=None):
        if status not in ('Created','Training','Paused','Completed','Interrupted','Failed','Archived','Stopped','Review'):raise ValueError('Unknown lifecycle state')
        with self.db(True) as db:
            if owner and db.execute("SELECT 1 FROM leases WHERE generation_id=? AND kind='training' AND token!=?",(generation,owner)).fetchone():return
            db.execute("UPDATE generations SET status=? WHERE id=? AND status!='Deleted'",(status,generation))

    def progress(self,generation,episodes,steps):
        with self.db(True) as db:
            row=db.execute('SELECT data FROM generations WHERE id=?',(generation,)).fetchone()
            if not row:return
            info=json.loads(row[0]);info.update(episode=max(info.get('episode',0),episodes),environment_steps=max(info.get('environment_steps',0),steps))
            db.execute('UPDATE generations SET data=? WHERE id=?',(json.dumps(info),generation))

    def register_run(self,run,family,path=None,interface='Frontier',generation=None,resume=None,description='',data=None):
        if resume:
            cp=self.find_checkpoint(resume)
            if not cp:raise ValueError('Resume checkpoint has no resolved generation; refresh the library')
            if generation and generation!=cp['generation_id']:raise ValueError('Resume cannot change generation identity')
            generation=cp['generation_id']
        with self.db(True) as db:
            old=db.execute('SELECT generation_id FROM aliases WHERE origin=?',('run:'+run,)).fetchone()
            if old:
                if generation and generation!=old[0]:raise ValueError('Run identity already belongs to another generation')
                generation=old[0]
            if generation:
                row=db.execute('SELECT family,status FROM generations WHERE id=?',(generation,)).fetchone()
                if not row or row[0]!=family or row[1] in ('Deleted','Deleting'):raise ValueError('Incompatible or deleted generation')
                db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?)',('run:'+run,generation))
            else:generation=self._new(db,family,'run:'+run,description,data=data)
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?) ON CONFLICT(run) DO UPDATE SET root=COALESCE(excluded.root,sessions.root)',
                (run,generation,str(Path(path).resolve()) if path else None,interface,'Created'))
            info=json.loads(db.execute('SELECT data FROM generations WHERE id=?',(generation,)).fetchone()[0])
            interfaces=info.setdefault('interfaces',[])
            if interface not in interfaces:interfaces.append(interface)
            if data and data.get('training'):
                info.setdefault('initial_training',data['training']);info['training']=data['training']
            db.execute('UPDATE generations SET data=? WHERE id=?',(json.dumps(info),generation))
        return generation

    def checkpoint(self,identifier):
        with self.db() as db:row=db.execute('SELECT * FROM checkpoints WHERE id=?',(identifier,)).fetchone()
        if not row:raise ValueError('Unknown checkpoint identity')
        item=dict(zip(('id','generation_id','origin','path','kind','episode','steps','status','data'),row));item['data']=json.loads(item['data'])
        return item

    def checkpoints(self,generation=None,include_deleted=False):
        with self.db() as db:ids=[r[0] for r in db.execute('SELECT id FROM checkpoints ORDER BY episode DESC,id')]
        items=[]
        for identifier in ids:
            cp=self.checkpoint(identifier)
            if generation and cp['generation_id']!=generation:continue
            if not include_deleted and (cp['status']=='Deleted' or self.get(cp['generation_id'])['status']=='Deleted'):continue
            path=Path(cp['path'])
            if cp['status'] not in ('Deleted','Deleting','Review','Corrupt','Superseded'):
                cp['status']='Available' if path.exists() else 'Missing'
                if cp['status']=='Available':
                    try:
                        header=metadata(path.with_suffix('.json') if path.is_file() else path/'training.json' if (path/'training.json').exists() else path/'network.json')
                        if family_of(header)!=self.get(cp['generation_id'])['family']:cp['status']='Incompatible'
                        elif header.get('sha256')!=cp['data'].get('sha256'):cp['status']='Review'
                    except (ValueError,OSError,KeyError,TypeError):cp['status']='Corrupt'
            if cp['kind']=='resumable' and not (path/'training.pt').is_file():cp['kind']='inference'
            items.append(cp)
        return items

    def find_checkpoint(self,path):
        path=str(Path(path).resolve())
        with self.db() as db:
            row=db.execute("SELECT id FROM checkpoints WHERE path=? ORDER BY CASE WHEN status IN ('Deleted','Superseded') THEN 1 ELSE 0 END,id DESC",(path,)).fetchone()
            if not row:row=db.execute('SELECT checkpoint_id FROM locations WHERE path=?',(path,)).fetchone()
        return self.checkpoint(row[0]) if row else None

    def generation_for_run(self,run):
        with self.db() as db:row=db.execute('SELECT generation_id FROM aliases WHERE origin=?',('run:'+run,)).fetchone()
        return self.get(row[0]) if row else None

    def index_checkpoint(self,path,data=None,generation=None):
        path=Path(path).resolve();published=path.is_file() and path.suffix=='.pt'
        data=data or metadata(path.with_suffix('.json') if published else path/'training.json' if (path/'training.json').exists() else path/'network.json')
        family=family_of(data)
        card=metadata(path/'model_card.json') if path.is_dir() and (path/'model_card.json').is_file() else {}
        run=data.get('model_identifier') or data.get('experiment_id') or card.get('model_identifier')
        if path.is_dir() and (path/'training.json').exists() and run!=path.parent.name:raise ValueError('Checkpoint location and run identity disagree; review required')
        source=card.get('source_checkpoint') or path.name
        origin=('run:'+run+':checkpoint:'+source) if run else 'import:'+data.get('sha256',hashlib.sha256(str(path).encode()).hexdigest())
        if generation is None:
            if run:
                known=self.generation_for_run(run)
                generation=known['id'] if known else self.register_run(run,family,interface=data.get('training_interface','Unknown'),description='',data={'historical_name':data.get('model_name') or card.get('model_name'),
                    'created_utc':data.get('created_utc'),'training':data.get('hyperparameters',card.get('hyperparameters'))})
            else:
                with self.db(True) as db:generation=self._new(db,family,origin,description='Imported model; training provenance unavailable')
                self.status(generation,'Review')
        if self.get(generation)['family']!=family:raise ValueError('Checkpoint family does not match generation')
        kind='resumable' if data.get('kind')=='resumable-boundary' and path.is_dir() and (path/'training.pt').is_file() else 'inference'
        with self.db(True) as db:
            old=db.execute('SELECT id,path,kind,status,data FROM checkpoints WHERE origin=?',(origin,)).fetchone()
            if old:
                if old[3]=='Deleted':return old[0]
                db.execute('INSERT INTO locations VALUES (?,?) ON CONFLICT(path) DO UPDATE SET checkpoint_id=excluded.checkpoint_id',(str(path),old[0]))
                # An exported copy never replaces a real recovery checkpoint.
                if old[2]=='resumable' and kind!='resumable':return old[0]
                if Path(old[1]).exists() and old[1]!=str(path):return old[0]
                identifier=old[0]
                if json.loads(old[4]).get('sha256')!=data.get('sha256'):
                    db.execute("UPDATE checkpoints SET status='Review' WHERE id=?",(identifier,))
                    return identifier
                db.execute('UPDATE checkpoints SET path=?,data=?,status=? WHERE id=?',(str(path),json.dumps(data),'Available',identifier))
            else:
                db.execute("UPDATE checkpoints SET status='Superseded' WHERE path=? AND status!='Deleted'",(str(path),))
                identifier=uuid4().hex
                db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?)',(identifier,generation,origin,str(path),kind,
                    data.get('episode',card.get('episode')),data.get('environment_steps',card.get('environment_steps')),'Available',json.dumps(data)))
                db.execute('INSERT INTO locations VALUES (?,?) ON CONFLICT(path) DO UPDATE SET checkpoint_id=excluded.checkpoint_id',(str(path),identifier))
            generation_row=db.execute('SELECT data,status FROM generations WHERE id=?',(generation,)).fetchone()
            info=json.loads(generation_row[0]);info.update(episode=max(info.get('episode',0),data.get('episode',card.get('episode',0)) or 0),
                environment_steps=max(info.get('environment_steps',0),data.get('environment_steps',card.get('environment_steps',0)) or 0))
            lifecycle=generation_row[1]
            if lifecycle in ('Created','Interrupted') and info['episode']:
                settings=data.get('hyperparameters') or info.get('training') or {}
                lifecycle='Completed' if info['episode']>=settings.get('episodes',10**12) else 'Interrupted'
            db.execute('UPDATE generations SET data=?,status=? WHERE id=?',(json.dumps(info),lifecycle,generation))
        return identifier

    def resolve(self,identifier,purpose='inference',verify=True):
        cp=self.checkpoint(identifier);generation=self.get(cp['generation_id'])
        if cp['status'] in ('Deleted','Deleting','Review','Superseded') or generation['status'] in ('Deleted','Deleting','Restoring'):raise ValueError('Selected checkpoint is unavailable; refresh the library')
        path=Path(cp['path'])
        if not path.exists():raise ValueError('Selected checkpoint is missing; no replacement was selected')
        if purpose in ('resume','legacy_resume') and cp['kind']!='resumable':raise ValueError('Inference-only checkpoint cannot resume training')
        if purpose in ('resume','legacy_resume') and not (path/'training.pt').is_file():raise ValueError('Training state is missing; only inference is available')
        if purpose=='legacy_resume' and (generation['family']!='Alpha' or cp['data'].get('parallel_execution')):raise ValueError('Use Frontier to resume registered or parallel models; Legacy supports single-environment Alpha')
        data=metadata(path.with_suffix('.json') if path.is_file() else path/'network.json')
        envelope=metadata(path/'training.json') if path.is_dir() and (path/'training.json').is_file() else data
        if parse_model_config(envelope)!=parse_model_config(cp['data']):raise ValueError('Selected checkpoint metadata changed; refresh the library')
        for key in ('sha256','model_identifier','experiment_id','episode','generation_id'):
            if cp['data'].get(key)!=envelope.get(key):raise ValueError('Selected checkpoint identity changed; no substitution was made')
        if family_of(data)!=generation['family']:raise ValueError('Checkpoint architecture no longer matches its generation')
        if purpose=='lab' and generation['family']!='Alpha':raise ValueError('The existing Neural Laboratory supports Alpha / 11-input models only')
        weights=path if path.is_file() else path/data['weights_file']
        expected_parent=path.parent if path.is_file() else path
        if weights.is_symlink() or weights.resolve().parent!=expected_parent:
            raise ValueError('Unsafe model weights path')
        if not weights.is_file():raise ValueError('Selected model weights are missing')
        if verify and hashlib.sha256(weights.read_bytes()).hexdigest()!=data.get('sha256'):
            with self.db(True) as db:db.execute("UPDATE checkpoints SET status='Corrupt' WHERE id=?",(identifier,))
            raise ValueError('Selected model weights are corrupted')
        return cp

    def load(self,identifier,purpose='inference'):
        cp=self.resolve(identifier,purpose)
        path=Path(cp['path'])
        if path.is_file():
            from ai.frontier_store import load_published_model
            network,data=load_published_model(path)
        elif (path/'training.json').is_file():
            from ai.checkpoints import CheckpointStore
            network,data=CheckpointStore(path.parent.parent).load_model(path)
        else:
            from ai.persistence import ModelStore
            network,data=ModelStore(path.parent).load(path.name)
        data={**data,'generation_id':cp['generation_id'],'library_checkpoint_id':cp['id'],
            'model_name':self.get(cp['generation_id'])['name'],'model_identifier':data.get('model_identifier',cp['generation_id']),
            'episode':cp['episode'] or 0}
        training=data.get('hyperparameters') or self.get(cp['generation_id'])['data'].get('training')
        if training:data['hyperparameters']=training
        return network,data

    def scan(self):
        """Index metadata only. Source files, names and historical journals stay intact."""
        result={'generations_added':0,'checkpoints_added':0,'review':list(getattr(self,'recovery_issues',[]))}
        before=len(self.generations(include_deleted=True));before_cp=len(self.checkpoints(include_deleted=True))
        roots=checkpoint_roots(self.profile/'checkpoints')
        # Legacy parents must be known before Frontier continuations are indexed.
        for path in sorted((self.profile/'checkpoints').glob('*/*/training.json')):
            try:self.index_checkpoint(path.parent)
            except (ValueError,OSError,KeyError,TypeError) as error:result['review'].append(str(path)+': '+str(error))
        experiments=[]
        for index in (self.profile/'frontier_index').glob('*.json'):
            try:experiments.append(Path(metadata(index)['experiment_root']))
            except (ValueError,OSError,KeyError):result['review'].append(str(index))
        if self.repository:experiments.extend(p.parent for p in (self.repository/'experiments/frontier').glob('*/experiment.json'))
        experiments.extend(p.parent for p in (self.profile/'frontier').glob('*/experiment.json'))
        pending=[]
        for folder in dict.fromkeys(experiments):
            try:
                data=metadata(folder/'experiment.json')
                pending.append((data.get('created_utc',''),folder,data))
            except (ValueError,OSError):result['review'].append(str(folder))
        # Index parent sessions before continuations; follow only evidenced checkpoint lineage.
        for _,folder,data in sorted(pending,key=lambda row:row[0]):
            run=data.get('experiment_id')
            try:
                if run!=folder.name:raise ValueError('Ambiguous experiment identity')
                cps=list((folder/'checkpoints'/run).glob('*/training.json'))
                if not cps:
                    if 'architecture' not in data:continue
                    envelope={'format_version':2 if 'configuration_id' in data['architecture'] else 1,**data}
                    family=family_of(envelope)
                else:family=family_of(metadata(cps[0]))
                resumed=data.get('resumed_from');parent=self.find_checkpoint(resumed) if resumed else None
                generation=self.register_run(run,family,folder,interface='Frontier',generation=parent['generation_id'] if parent else None,
                    data={'historical_name':data.get('name'),'created_utc':data.get('created_utc'),'training':data.get('training')})
                if resumed and not parent:
                    result['review'].append(f'{run}: unresolved resume parent; kept separate for review')
                    self.status(generation,'Review')
                for p in cps:self.index_checkpoint(p.parent,generation=generation)
                # Published pairs have inference-only capability, never fabricated recovery.
                for p in (folder/'models').glob('*.pt'):self.index_checkpoint(p,generation=generation)
            except (ValueError,OSError,KeyError,TypeError) as error:result['review'].append(str(folder)+': '+str(error))
        for base in roots:
            for path in sorted(base.glob('*/*/training.json')):
                try:self.index_checkpoint(path.parent)
                except (ValueError,OSError,KeyError,TypeError) as error:result['review'].append(str(path)+': '+str(error))
        for base in (self.profile/'models',*((self.repository/'models',) if self.repository else ())):
            for path in sorted(base.glob('*/network.json')):
                try:self.index_checkpoint(path.parent)
                except (ValueError,OSError,KeyError,TypeError) as error:result['review'].append(str(path)+': '+str(error))
        result['review'].extend(self._index_references())
        result['generations_added']=len(self.generations(include_deleted=True))-before
        result['checkpoints_added']=len(self.checkpoints(include_deleted=True))-before_cp
        return result

    def _index_references(self):
        issues=[]
        for path in (self.profile/'evaluations').glob('*.json'):
            try:
                data=metadata(path,50000000)
                owners=[self.find_checkpoint(model['checkpoint']) for model in data['models']]
                shared=any(cp is None for cp in owners) or len({cp['generation_id'] for cp in owners if cp})>1
                if any(cp is None for cp in owners):issues.append('Unresolved evaluation lineage retained: '+str(path))
                for index,model in enumerate(data['models']):
                    cp=owners[index]
                    if cp:
                        with self.db(True) as db:db.execute('INSERT INTO references_data VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind',
                            (str(path.resolve())+':'+str(index),cp['generation_id'],cp['id'],str(path.resolve()),'evaluation_shared' if shared else 'evaluation'))
            except (ValueError,OSError,KeyError,TypeError):continue
        from ai.history import HistoryStore
        history=HistoryStore(self.profile/'history')
        try:
            for run in history.datasets():
                origin=run['configuration'].get('checkpoint_run_id',run['identifier'])
                with self.db(True) as db:
                    owner=db.execute('SELECT generation_id FROM aliases WHERE origin=?',('run:'+origin,)).fetchone()
                    if owner:db.execute('INSERT OR IGNORE INTO references_data VALUES (?,?,?,?,?)',
                        ('history:'+run['identifier'],owner[0],None,str(history.path),'history'))
        except (ValueError,OSError,sqlite3.Error):pass
        for path in (self.profile/'experiments').glob('*.json'):
            try:
                data=metadata(path)
                owners=[self.generation_for_run(str(run).split(':session:')[0]) for run in data.get('training_runs',[])]
                if not owners or any(owner is None for owner in owners):issues.append('Unassigned experiment metadata retained: '+str(path))
                shared=not owners or any(owner is None for owner in owners) or len({owner['id'] for owner in owners if owner})>1
                for owner in (item for item in owners if item):
                    for file in (path,path.with_suffix('.events.jsonl')):
                        if file.is_file():
                            with self.db(True) as db:db.execute('INSERT OR REPLACE INTO references_data VALUES (?,?,?,?,?)',
                                ('experiment:'+str(file.resolve())+':'+owner['id'],owner['id'],None,str(file.resolve()),'experiment_shared' if shared else 'experiment'))
            except (ValueError,OSError,KeyError,TypeError):continue
        for path in (self.profile/'exports').glob('*.metadata.json'):
            try:
                data=metadata(path)
                owners=[self.find_checkpoint(item['checkpoint']) if item.get('checkpoint') else None for item in data['datasets']]
                if not owners or any(owner is None for owner in owners):issues.append('Unassigned export metadata retained: '+str(path))
                shared=not owners or any(owner is None for owner in owners) or len({owner['generation_id'] for owner in owners if owner})>1
                for owner in (item for item in owners if item):
                    for file in (path,Path(str(path)[:-len('.metadata.json')])):
                        if file.is_file():
                            with self.db(True) as db:db.execute('INSERT OR REPLACE INTO references_data VALUES (?,?,?,?,?)',
                                ('export:'+str(file.resolve())+':'+owner['generation_id'],owner['generation_id'],owner['id'],str(file.resolve()),'export_shared' if shared else 'export'))
            except (ValueError,OSError,KeyError,TypeError):continue
        return issues

    def references(self,generation):
        with self.db() as db:return [dict(zip(('id','generation_id','checkpoint_id','path','kind'),row)) for row in db.execute('SELECT * FROM references_data WHERE generation_id=?',(generation,))]

    def _safe_path(self,path):
        path=Path(path)
        with self.db() as db:roots=[Path(row[0]) for row in db.execute('SELECT path FROM managed_roots')]
        resolved=path.resolve()
        if not any(resolved!=base and resolved.is_relative_to(base) for base in roots):raise ValueError('Files are outside managed storage; deletion is blocked')
        for parent in (path,*path.parents):
            if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('Deletion cannot follow a symbolic link or junction')
        if path.is_dir():
            for base,dirs,files in os.walk(path,followlinks=False):
                for name in dirs+files:
                    child=Path(base)/name
                    if child.is_symlink() or getattr(child,'is_junction',lambda:False)():raise ValueError('Linked files require manual review')
        if path.drive.lower()!=self.root.drive.lower():raise ValueError('Cross-volume Trash is unsupported; files were not changed')
        return resolved

    def _protected(self,db,generation,checkpoint=None):
        lease=db.execute('SELECT owner FROM leases WHERE generation_id=?'+(' AND (checkpoint_id=? OR kind=\'training\')' if checkpoint else ''),
            (generation,checkpoint) if checkpoint else (generation,)).fetchone()
        if lease:raise ValueError('Deletion blocked: active or loaded by '+lease[0])
        for row in db.execute('SELECT root FROM sessions WHERE generation_id=?',(generation,)):
            if row[0] and (Path(row[0])/'.writer.lock').exists():raise ValueError('Deletion blocked by a writer lock; close the training session')

    def preview_delete(self,kind,identifier):
        if kind not in ('generation','checkpoint'):raise ValueError('Unknown deletion type')
        self._expire_leases()
        cp=self.checkpoint(identifier) if kind=='checkpoint' else None
        generation=cp['generation_id'] if cp else identifier
        item=cp or self.get(identifier)
        if item['status'] in ('Deleted','Deleting'):raise ValueError('Item already deleted or being deleted')
        paths=[];shared=[]
        with self.db() as db:
            self._protected(db,generation,identifier if cp else None)
            if cp:
                paths=[] if cp['status']=='Superseded' else [Path(cp['path'])]
                if paths:paths.extend(Path(r[0]) for r in db.execute('SELECT path FROM locations WHERE checkpoint_id=?',(cp['id'],)))
                if paths and paths[0].is_file():paths.append(paths[0].with_suffix('.json'))
            else:
                for (folder,) in db.execute('SELECT root FROM sessions WHERE generation_id=?',(generation,)):
                    if folder:
                        folder=Path(folder)
                        if (folder/'experiment.json').is_file():
                            experiment=metadata(folder/'experiment.json')
                            owner=db.execute('SELECT generation_id FROM aliases WHERE origin=?',('run:'+experiment.get('experiment_id',''),)).fetchone()
                            if owner and owner[0]==generation:paths.append(folder)
                paths.extend(Path(r[0]) for r in db.execute("SELECT path FROM checkpoints WHERE generation_id=? AND status NOT IN ('Deleted','Superseded')",(generation,)))
                paths.extend(Path(r[0]) for r in db.execute("SELECT locations.path FROM locations JOIN checkpoints ON checkpoints.id=locations.checkpoint_id WHERE checkpoints.generation_id=? AND checkpoints.status NOT IN ('Deleted','Superseded')",(generation,)))
                paths.extend(p.with_suffix('.json') for p in list(paths) if p.is_file() and p.suffix=='.pt')
                refs=db.execute('SELECT DISTINCT path,kind FROM references_data WHERE generation_id=?',(generation,)).fetchall()
                for path,ref_kind in refs:
                    other=db.execute('SELECT 1 FROM references_data WHERE path=? AND generation_id!=?',(path,generation)).fetchone()
                    if other or ref_kind not in ('evaluation','experiment','export'):shared.append(path)
                    else:paths.append(Path(path))
            unique=[]
            for path in sorted(set(paths),key=lambda p:len(p.parts)):
                if not path.exists():continue
                path=self._safe_path(path)
                if any(path==parent or path.is_relative_to(parent) for parent in unique):continue
                # Never move a resource indexed to another generation.
                for other_path,other_gen in db.execute("SELECT path,generation_id FROM checkpoints WHERE status!='Deleted'"):
                    if other_gen!=generation and (Path(other_path)==path or Path(other_path).is_relative_to(path)):raise ValueError('Shared ownership requires review; deletion blocked')
                for other_path,other_gen in db.execute('SELECT path,generation_id FROM references_data'):
                    if other_gen!=generation and (Path(other_path)==path or Path(other_path).is_relative_to(path)):raise ValueError('Shared records inside this directory require review; deletion blocked')
                unique.append(path)
        files=[str(f) for path in unique for f in (path.rglob('*') if path.is_dir() else (path,)) if f.is_file()]
        plan=dict(kind=kind,id=identifier,generation_id=generation,paths=[str(p) for p in unique],files=files,retained_shared=shared,previous=item['status'])
        plan['token']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        return plan

    def delete(self,preview):
        current=self.preview_delete(preview['kind'],preview['id'])
        if current!=preview:raise ValueError('Deletion preview is stale; review the new preview')
        operation=uuid4().hex
        moves=[(path,str(self.root/'Trash'/operation/str(index))) for index,path in enumerate(preview['paths'])]
        with self.db(True) as db:
            self._protected(db,preview['generation_id'],preview['id'] if preview['kind']=='checkpoint' else None)
            table='generations' if preview['kind']=='generation' else 'checkpoints'
            db.execute(f"UPDATE {table} SET status='Deleting' WHERE id=?",(preview['id'],))
            db.execute('INSERT INTO trash VALUES (?,?,?,?,?,?)',(operation,preview['kind'],preview['id'],'Preparing',preview['previous'],json.dumps(moves)))
        try:
            for source,destination in moves:
                self._safe_path(source);target=self._trash_path(destination,operation);target.parent.mkdir(parents=True,exist_ok=True)
                os.replace(source,target)
            with self.db(True) as db:
                db.execute(f"UPDATE {table} SET status='Deleted' WHERE id=?",(preview['id'],))
                db.execute("UPDATE trash SET state='Trashed' WHERE id=?",(operation,))
            return operation
        except BaseException:
            self._rollback_operation(operation)
            raise

    def _rollback_operation(self,identifier):
        with self.db() as db:row=db.execute('SELECT kind,target,previous,plan FROM trash WHERE id=?',(identifier,)).fetchone()
        if not row:return
        kind,target,previous,plan=row
        for source,destination in reversed(json.loads(plan)):
            self._trash_path(destination,identifier)
            if Path(destination).exists():
                if Path(source).exists():raise ValueError('Trash recovery conflict; manual review required')
                self._safe_path(source);Path(source).parent.mkdir(parents=True,exist_ok=True);os.replace(destination,source)
        with self.db(True) as db:
            db.execute(f'UPDATE {"generations" if kind=="generation" else "checkpoints"} SET status=? WHERE id=?',(previous,target))
            db.execute("UPDATE trash SET state='Cancelled' WHERE id=?",(identifier,))

    def _trash_path(self,path,operation):
        path=Path(path);base=self.root/'Trash'/operation
        if not path.resolve().is_relative_to(base):raise ValueError('Unsafe Trash destination')
        for parent in (path,*path.parents):
            if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('Trash cannot follow linked paths')
        return path

    def _recover_operations(self):
        self.recovery_issues=[]
        with self.db() as db:pending=[r[0] for r in db.execute("SELECT id FROM trash WHERE state IN ('Preparing','Restoring')")]
        for operation in pending:
            try:self._rollback_operation(operation)
            except (ValueError,OSError) as error:self.recovery_issues.append('Trash recovery '+operation+': '+str(error))

    def trashed(self):
        with self.db() as db:return [dict(zip(('id','kind','target','state','previous','plan'),row)) for row in db.execute("SELECT * FROM trash WHERE state='Trashed'")]

    def restore(self,operation):
        with self.db(True) as db:
            row=db.execute("SELECT kind,target,previous,plan FROM trash WHERE id=? AND state='Trashed'",(operation,)).fetchone()
            if not row:raise ValueError('Trash item is no longer available')
            kind,target,previous,plan=row
            for source,destination in json.loads(plan):
                self._safe_path(source)
                if Path(source).exists():raise ValueError('Restore would overwrite existing files')
                self._trash_path(destination,operation)
                if not Path(destination).exists():raise ValueError('Trash files are missing; restore was not performed')
            db.execute("UPDATE trash SET state='Restoring' WHERE id=?",(operation,))
        self._rollback_operation(operation)
