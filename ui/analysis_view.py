"""First-class, background-driven training/evaluation analysis workspace."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
import pygame
from ai.analysis_repository import AnalysisRepository
from ai.analysis_model import AnalysisModel
from ai.analysis_export import export_csv
from ai.evaluation import evaluate_checkpoints
from ai.experiments import Experiment,ExperimentStore,BASELINE_ENCODER
from ui.ai_components import FormView
from ui.analysis_chart import AnalysisChart,MODEL_COLORS
from ui.components import BG,PANEL,TEXT,MUTED,GREEN,label


class AnalysisView(FormView):
    def __init__(self,profile):
        super().__init__('train:analysis:')
        self.profile=Path(profile)
        self.repository=AnalysisRepository(profile)
        self.model=AnalysisModel()
        self.chart=AnalysisChart()
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='snake-analysis')
        self.jobs={}
        self.waiting={}
        self.cancel=threading.Event()
        self.catalog=()
        self.indices=[0,None]
        self.datasets=[None,None]
        self.mode='distribution'
        self.group=100
        self.alignment='episode'
        self.ranges={}
        self.show_mean=self.show_median=True
        self.full_export=False
        self.bundle=None
        self.generation=0
        self.stats_generation=0
        self.text={'start':'1','end':'10000','games':'25','seed':'0','step_bin':'25000'}
        self.status='Select Model A; Model B is optional. Loading is read-only.'
        self.progress=''
        self.signatures=None
        self.refresh_at=0.
        self.last_experiment_id=None
        self.layout()

    def submit(self,key,operation,*args,callback=None):
        if key in self.jobs:
            self.waiting[key]=(operation,args,callback)
            return
        self.jobs[key]=(self.pool.submit(operation,*args),callback)

    def refresh(self):
        self.submit('catalog',self.repository.choices,callback=self.catalog_ready)
        self.refresh_at=time.monotonic()

    def catalog_ready(self,catalog):
        old_ids=[self.catalog[i].identifier if i is not None and i<len(self.catalog) else None for i in self.indices]
        self.catalog=catalog
        for slot,identifier in enumerate(old_ids):
            if identifier:
                self.indices[slot]=next((i for i,c in enumerate(catalog) if c.identifier==identifier),None)
        if self.indices[0] is None and catalog:self.indices[0]=0
        if catalog:self.load_selected()
        else:self.status='No historical runs yet. Complete an episode in Training Center or load an existing profile.'
        self.layout()

    def load_selected(self):
        choices=tuple(self.catalog[i] for i in self.indices if i is not None and i<len(self.catalog))
        self.generation+=1
        self.stats_generation+=1
        self.bundle=None
        self.status='Loading exact raw records in background'
        generation=self.generation
        def load():return tuple(self.repository.load(c) for c in choices)
        def ready(data):
            if generation!=self.generation:return
            self.datasets=list(data)+[None]*(2-len(data))
            if not self.ranges:
                records=[r for d in data for r in d.records]
                if records:
                    self.text['start'],self.text['end']=str(min(r.episode for r in records)),str(max(r.episode for r in records))
            self.ranges[self.alignment]=(int(self.text['start']),int(self.text['end']))
            self.prepare()
        self.submit('load',load,callback=ready)

    def prepare(self):
        datasets=tuple(d for d in self.datasets if d is not None)
        if not datasets:return
        start,end=int(self.text['start']),int(self.text['end'])
        if start<0 or end<start:raise ValueError('Range must have start >= 0 and end >= start')
        self.stats_generation+=1
        self.bundle=None
        generation=self.stats_generation
        def ready(bundle):
            if generation==self.stats_generation:
                self.bundle=bundle
                self.status=('Frozen evaluation: compare only identical seed sets/settings.' if datasets[0].kind=='evaluation'
                    else 'Training includes epsilon exploration; these curves are not definitive policy rankings.')
                self.layout()
        self.submit('statistics',self.model.prepare,datasets,self.mode,self.group,self.alignment,start,end,int(self.text['step_bin']),callback=ready)

    def layout(self):
        self.buttons,self.fields=[],{}
        for slot,x in enumerate((24,640)):
            selected=self.catalog[self.indices[slot]] if self.indices[slot] is not None and self.indices[slot]<len(self.catalog) else None
            title=f"MODEL {'AB'[slot]}: "+(selected.name if selected else 'OPTIONAL / NONE' if slot else 'No selection')
            self.button(title[:72],x,113,600,'model_a' if slot==0 else 'model_b')
        for x,width,title,action in ((24,174,'DISTRIBUTION','distribution'),(207,174,'MEAN / MEDIAN','mean_median'),
            (390,132,'RAW SCORES','raw'),(532,130,f'WINDOW {self.group}','group'),
            (672,170,'BY '+('EPISODE' if self.alignment=='episode' else 'ENV STEPS'),'alignment'),
            (852,110,'MEAN '+('ON' if self.show_mean else 'OFF'),'mean'),
            (972,130,'MEDIAN '+('ON' if self.show_median else 'OFF'),'median'),(1110,146,'REFRESH','refresh')):
            self.button(title,x,222,width,action,self.mode==action)
        self.field('start',79,600,100)
        self.field('end',222,600,100)
        self.button('APPLY',332,599,77,'range')
        for x,title,action in ((419,'ZOOM +','zoom_in'),(512,'ZOOM -','zoom_out'),(605,'<','left'),(664,'>','right'),(723,'RESET ZOOM','reset')):
            self.button(title,x,599,105 if action=='reset' else 83 if action.startswith('zoom') else 49,action)
        self.field('step_bin',835,600,76)
        self.button('EXPORT CSV: RAW',24,650,183,'export_raw')
        self.button('CSV: GROUPED',217,650,169,'export_grouped')
        self.button('FULL' if self.full_export else 'RANGE',396,650,87,'full')
        self.button('EVALUATE FROZEN',493,650,190,'evaluate')
        self.field('games',735,652,76)
        self.field('seed',867,652,84)
        self.button('25/100/500',961,650,130,'preset')
        self.button('EXPERIMENT A',1101,650,155,'experiment')

    def bounds(self):
        values=[]
        for dataset in self.datasets:
            if dataset:
                for record in dataset.records:
                    value=record.episode if self.alignment=='episode' else record.cumulative_steps
                    if value is None:raise ValueError('Recorded cumulative steps unavailable; use episode alignment')
                    values.append(value)
        return (min(values),max(values)) if values else (1,1)

    def action(self,action):
        try:
            if action in ('model_a','model_b'):
                if not self.catalog:return
                slot=int(action=='model_b')
                index=self.indices[slot]
                self.indices[slot]=(index+1 if index is not None else 0)%len(self.catalog)
                if slot==1 and index==len(self.catalog)-1:self.indices[slot]=None
                self.load_selected()
            elif action=='refresh':self.refresh()
            elif action in ('distribution','mean_median','raw'):
                self.mode=action
                self.prepare()
            elif action=='group':
                options=(25,50,100,250,500)
                self.group=options[(options.index(self.group)+1)%len(options)]
                self.prepare()
            elif action=='alignment':
                self.ranges[self.alignment]=(int(self.text['start']),int(self.text['end']))
                previous=self.alignment
                self.alignment='steps' if previous=='episode' else 'episode'
                try:bounds=self.bounds()
                except ValueError:
                    self.alignment=previous
                    raise
                start,end=self.ranges.get(self.alignment,bounds)
                self.text['start'],self.text['end']=str(start),str(end)
                self.prepare()
            elif action in ('mean','median'):
                setattr(self,'show_'+action,not getattr(self,'show_'+action))
            elif action in ('zoom_in','zoom_out','left','right','reset','range'):
                start,end=int(self.text['start']),int(self.text['end'])
                low,high=self.bounds()
                span=max(1,end-start)
                if action=='reset':start,end=low,high
                elif action.startswith('zoom'):
                    span=max(2,int(span*(.5 if action=='zoom_in' else 2)))
                    center=(start+end)//2
                    start,end=max(low,center-span//2),min(high,center+span//2)
                elif action in ('left','right'):
                    shift=max(1,span//3)*(1 if action=='right' else -1)
                    start=max(low,min(start+shift,max(low,high-span)))
                    end=min(high,start+span)
                self.text['start'],self.text['end']=str(start),str(end)
                self.ranges[self.alignment]=(start,end)
                self.prepare()
            elif action=='full':self.full_export=not self.full_export
            elif action.startswith('export_'):
                datasets=tuple(d for d in self.datasets if d)
                if not datasets:raise ValueError('Select data before exporting')
                kind='raw' if action=='export_raw' else 'grouped'
                path=self.profile/'exports'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8]+'-'+kind+'.csv')
                start,end=(None,None) if self.full_export else (int(self.text['start']),int(self.text['end']))
                size=self.group if self.alignment=='episode' else int(self.text['step_bin'])
                self.submit('export',export_csv,path,datasets,kind,start,end,size,self.alignment,
                            callback=lambda p:setattr(self,'status','Exported '+str(p)))
                self.status='Exporting in background; existing files are never overwritten'
            elif action=='preset':
                options=(25,100,500)
                current=int(self.text['games'])
                self.text['games']=str(options[(options.index(current)+1)%3] if current in options else 25)
            elif action=='evaluate':
                if 'evaluation' in self.jobs and not self.jobs['evaluation'][0].done():
                    self.cancel.set()
                    self.status='Canceling evaluation safely'
                    return
                choices=[self.catalog[i] for i in self.indices if i is not None]
                if not choices or any(c.checkpoint is None for c in choices):raise ValueError('Select saved checkpoints before evaluation')
                n,seed=int(self.text['games']),int(self.text['seed'])
                if not 1<=n<=10000 or not 0<=seed<2**63-n:raise ValueError('Choose 1..10,000 games and a valid seed start')
                self.cancel.clear()
                identifiers={self.last_experiment_id} if self.last_experiment_id else set()
                identifiers.update(d.configuration.get('experiment_id') for d in self.datasets if d and d.configuration and d.configuration.get('experiment_id'))
                def progress(model,game,total):self.progress=f'Evaluating model {model+1}: {game}/{total}'
                def ready(path):
                    self.status='Frozen evaluation saved: '+str(path.name)+' / select EVAL results in Model A/B'
                    for identifier in identifiers:
                        ExperimentStore(self.profile/'experiments').link(identifier,'evaluation',path)
                    self.refresh()
                self.submit('evaluation',evaluate_checkpoints,[c.checkpoint for c in choices],tuple(range(seed,seed+n)),
                    self.profile/'evaluations',self.cancel,progress,callback=ready)
                self.status='Evaluation uses identical seeds and the first checkpoint’s explicit termination settings'
            elif action=='experiment':
                dataset=self.datasets[0]
                if dataset is None:raise ValueError('Select Model A first')
                if dataset.architecture[0]!=11:raise ValueError('Only Experiment A / baseline 11 inputs is executable in v1.5')
                choices=[self.catalog[i] for i in self.indices if i is not None]
                configuration=dataset.configuration or {}
                config=configuration.get('training',configuration)
                rewards={key:config.get(key) for key in ('food_reward','collision_reward','step_reward','stagnation_reward')}
                n,seed=int(self.text['games']),int(self.text['seed'])
                if not 1<=n<=10000 or not 0<=seed<2**63-n:raise ValueError('Invalid evaluation seed set')
                experiment=Experiment(uuid4().hex,'Experiment A — '+dataset.name,BASELINE_ENCODER,dataset.architecture[0],
                    dataset.architecture,config,rewards,config.get('seed',0),tuple(range(seed,seed+n)),
                    config.get('episodes',1)*config.get('episode_cap',10000),tuple(c.run for c in choices),
                    tuple(str(c.checkpoint) for c in choices if c.checkpoint),())
                path=ExperimentStore(self.profile/'experiments').save(experiment)
                self.last_experiment_id=experiment.identifier
                self.status='Experiment metadata saved: '+path.name+' / budget is a declared plan, not a training change'
            self.layout()
        except (ValueError,OSError,RuntimeError,TypeError) as error:
            self.status=str(error)

    def update(self,dt,active=True):
        for key,(future,callback) in list(self.jobs.items()):
            if not future.done():continue
            del self.jobs[key]
            queued=self.waiting.pop(key,None)
            try:
                result=future.result()
                if callback:callback(result)
            except Exception as error:self.status=str(error)
            if queued and key not in self.jobs:
                operation,args,queued_callback=queued
                self.submit(key,operation,*args,callback=queued_callback)
        if active and time.monotonic()-self.refresh_at>3 and 'signature' not in self.jobs:
            self.refresh_at=time.monotonic()
            def changed(signature):
                if signature!=self.signatures:
                    self.signatures=signature
                    self.refresh()
            self.submit('signature',self.repository.signature,callback=changed)

    def handle_event(self,event,mouse):
        if event.type==pygame.MOUSEWHEEL and pygame.Rect(24,266,886,300).collidepoint(mouse):
            self.action('zoom_in' if event.y>0 else 'zoom_out')
            return True
        return super().handle_event(event,mouse)

    def close(self):
        self.cancel.set()
        self.waiting.clear()
        self.pool.shutdown(wait=True,cancel_futures=True)

    def draw(self,surface,mouse):
        for index,dataset in enumerate(self.datasets):
            if not dataset:continue
            x=24+index*616
            records=dataset.records
            steps=self.bundle['models'][index]['recorded_steps'] if self.bundle and index<len(self.bundle['models']) else None
            old_clip=surface.get_clip()
            surface.set_clip(pygame.Rect(x,150,600,64))
            label(surface,f"{'AB'[index]} / {dataset.kind.upper()} / {dataset.architecture} / {dataset.parameter_count:,} params",(x,154),13,MODEL_COLORS[index],False)
            completed=records[-1].episode if records else 0
            label(surface,f"Inputs {dataset.architecture[0]} / ep {completed:,} / n {len(records):,} / steps {steps if steps is not None else 'N/A'} / {dataset.status}"[:92],(x,175),13,TEXT,False)
            label(surface,(dataset.checkpoint or 'Durable raw episode journal')[-95:],(x,196),11,MUTED,False)
            surface.set_clip(old_clip)
        self.chart.draw(surface,pygame.Rect(24,266,886,300),self.bundle,mouse,self.show_mean,self.show_median)
        label(surface,'A emerald / B blue; mean solid, median dashed; box dot = mean. Dense views show exact sampled extrema/candles.',(24,573),11,MUTED,False)
        if self.bundle:
            pygame.draw.rect(surface,PANEL,(934,266,322,370),border_radius=6)
            label(surface,'STATISTICS / FULL RAW RANGE',(947,274),14,GREEN,False)
            keys=(('N','count'),('Mean','mean'),('Median','median'),('Min','minimum'),('Q1','q1'),('Q3','q3'),('Max','maximum'),
                ('IQR','iqr'),('SD (population)','stddev'),('Apples','total_apples'),('Mean length','mean_length'),('Median length','median_length'),
                ('Collision %','collision_percent'),('Stagnation %','stagnation_percent'),('Mean reward','mean_reward'),('Best episode','best_episode'),('Recorded steps','environment_steps'),('Duration (s)','duration'))
            label(surface,'A',(1114,303),12,MODEL_COLORS[0]);label(surface,'B',(1204,303),12,MODEL_COLORS[1])
            for row,(title,key) in enumerate(keys):
                label(surface,title,(946,322+row*16),12,MUTED,False)
                for index,model in enumerate(self.bundle['models']):
                    value=model['summary'][key]
                    text='N/A' if value is None else f'{value:.5g}'
                    label(surface,text,(1080+index*90,322+row*16),12,MODEL_COLORS[index],False)
        if self.chart.tooltip:
            pygame.draw.rect(surface,PANEL,(65,458,795,83),border_radius=6)
            for row,text in enumerate(self.chart.tooltip):label(surface,text,(78,468+row*22),14,TEXT,False)
        label(surface,'FROM',(24,607),12,MUTED,False);label(surface,'TO',(190,607),12,MUTED,False)
        label(surface,'STEP BIN',(835,584),10,MUTED,False)
        label(surface,'GAMES',(689,661),11,MUTED,False);label(surface,'SEED',(822,661),11,MUTED,False)
        self.draw_fields(surface)
        note=self.progress if 'evaluation' in self.jobs else self.status
        limits=[d.limitations[1] if len(d.limitations)>1 else d.limitations[0] for d in self.datasets if d and d.limitations]
        if limits and not self.jobs:note=limits[0]+' | '+self.status
        label(surface,note[:176],(24,697),11,MUTED,False)
