"""Frontier configuration and snapshot-driven headless training dashboard."""
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import time
import pygame
from ai.config import NetworkConfig
from ai.training_config import TrainingConfig
from ai.frontier_config import FrontierConfig
from ai.frontier import FrontierWorker
from ai.checkpoints import CheckpointStore
from ai.persistence import ModelStore
from ui.ai_components import FormView,chart
from ui.analysis_view import AnalysisView
from ui.analysis_chart import AnalysisChart
from ui.components import BG,PANEL,GREEN,TEXT,MUTED,label
from ai.model_registry import model_spec,LINEUP,identity,parse_model_config
from services.training_time import duration


class FrontierView(FormView):
    def __init__(self,profile,lab_provider,output):
        super().__init__('frontier:')
        self.profile=Path(profile)
        self.lab_provider=lab_provider
        self.store=CheckpointStore(self.profile/'checkpoints')
        self.models=ModelStore(self.profile/'models')
        self.worker=None
        self.snapshot=None
        self.page='config'
        self.advanced=False
        self.advanced_page=0
        self.source='new'
        self.model_configuration='A'
        self.selected=0
        self.refresh=100
        self.headless=True
        self.profiling=False
        self.environments=1
        self.strategy='auto'
        self.update_policy='transition'
        self.confirm=None
        self.library=[]
        self.library_at=0.
        self.text={key:str(value) for key,value in TrainingConfig().to_dict().items()}
        self.text.update(name='Snake DQN Frontier',hidden='16,8',activation='relu',recovery_interval='1000',recovery_slots='2',output=str(output))
        self.status='Choose collector count and policy. New experiments never overwrite existing runs.'
        self.analysis=AnalysisView(profile)
        self.analysis.prefix='frontier:analysis:'
        self.cache=None
        self.cache_key=None
        self.layout()

    def entries(self):
        if time.monotonic()-self.library_at>1:
            self.library=self.store.entries();self.library_at=time.monotonic()
        return self.library

    def configuration(self):
        defaults=TrainingConfig().to_dict()
        config=TrainingConfig(**{key:int(self.text[key]) if type(value) is int else float(self.text[key]) for key,value in defaults.items()})
        if self.source=='new' and self.model_configuration!='A':
            hidden=tuple(int(v.strip()) for v in self.text['hidden'].split(',')) if self.model_configuration=='B' else None
            if self.text['activation']!='relu':raise ValueError('Registered distance/CNN architectures use ReLU')
            architecture=model_spec(self.model_configuration,config.seed,hidden)
        else:architecture=NetworkConfig(tuple(int(v.strip()) for v in self.text['hidden'].split(',') if v.strip()) or (16,8),self.text['activation'],config.seed)
        execution=FrontierConfig(self.refresh,self.headless,int(self.text['recovery_interval']),int(self.text['recovery_slots']),
            profiling=self.profiling,environments=self.environments,strategy=self.strategy,update_policy=self.update_policy)
        if not 1<=len(self.text['name'].strip())<=40:raise ValueError('Model name must contain 1..40 characters')
        if not self.text['output'].strip():raise ValueError('Choose an experiment output directory')
        return config,architecture,execution

    def advanced_keys(self):
        keys=[k for k in TrainingConfig().to_dict() if k not in ('episodes','seed','checkpoint_interval')]
        if self.source=='new' and self.model_configuration in ('A','B'):keys+=['hidden','activation']
        return keys

    def model_info(self):
        if self.source=='new':return identity(self.configuration()[1])
        if self.source=='lab':return identity(self.lab_provider().network.config)
        if self.source=='resume':
            entries=self.entries()
            if not entries:return None
            return identity(parse_model_config(self.store.metadata(entries[self.selected%len(entries)]['path'])))
        metadata=json.loads((self.models.path(self.source)/'network.json').read_text(encoding='utf-8'))
        return identity(parse_model_config(metadata))

    def layout(self):
        self.buttons,self.fields=[],{}
        for x,title,action in ((24,'CONFIGURATION','config'),(251,'DASHBOARD','dashboard'),(478,'ANALYSIS','analysis')):
            self.button(title,x,67,215,action,self.page==action)
        self.button('TRAINING MENU',1060,67,196,'back')
        if self.page=='analysis':
            self.analysis.layout();self.buttons.extend(self.analysis.buttons)
            return
        if self.page=='config':
            if self.advanced:
                keys=self.advanced_keys()
                for index,key in enumerate(keys[self.advanced_page*12:(self.advanced_page+1)*12]):
                    column,row=index//6,index%6
                    self.field(key,24+column*620,199+row*60,580)
                self.button('BASIC',24,120,180,'advanced')
                self.button('ADVANCED PAGE '+str(self.advanced_page+1),220,120,300,'advanced_page')
            else:
                self.button('SOURCE: '+self.source,24,120,300,'source')
                title=self.model_configuration+': '+LINEUP[self.model_configuration][0] if self.source=='new' else 'ARCHITECTURE / ENCODER FROM SAVED MODEL'
                self.button(title,334,120,600,'model_configuration')
                self.button('ADVANCED DQN',960,120,270,'advanced')
                for key,x,y,width in (('name',24,210,580),('episodes',24,285,270),('seed',334,285,270),
                    ('recovery_interval',650,285,270),('recovery_slots',960,285,270),('output',24,457,1206)):
                    self.field(key,x,y,width)
                self.button(f'DASHBOARD EVERY {self.refresh} EPISODES',24,350,580,'refresh')
                self.button('HEADLESS '+('ON' if self.headless else 'PACED (9 steps/s)'),650,350,580,'headless')
                self.button('PROFILING '+('ON' if self.profiling else 'OFF'),650,402,580,'profiling')
                self.button('PARALLEL ENVIRONMENTS: '+str(self.environments),24,402,280,'environments')
                self.button('STRATEGY: '+self.strategy.upper(),314,402,290,'strategy')
                self.button('UPDATES: 1 PER '+('TRANSITION (REFERENCE)' if self.update_policy=='transition' else 'VECTOR STEP (FEWER UPDATES)'),24,510,600,'update_policy')
                entries=self.entries()
                if self.source=='resume':
                    entry=entries[self.selected%len(entries)] if entries else None
                    self.button('CHECKPOINT: '+(entry['name']+' / '+str(entry['episode'])+' ep' if entry else 'None available'),650,510,580,'select')
            self.button('START',24,635,220,'start',True)
        else:
            for x,title,action in ((24,'PAUSE','pause'),(190,'RESUME','resume'),(356,'STOP','stop'),(522,'SAVE CHECKPOINT','save'),(760,'OPEN IN ARENA','arena')):
                self.button(title,x,650,220 if action in ('save','arena') else 150,action)
            self.button('OPEN EXPERIMENT FILES',1000,650,256,'files')
        if self.confirm:
            partial=self.snapshot and self.snapshot.metrics.get('partial_episode_recovery')
            self.button('SAVE PARTIAL EPISODES / STOP' if partial else 'FINISH EPISODE & SAVE / STOP',235,405,390,'save_stop',True)
            self.button('STOP / KEEP IN MEMORY',650,405,350,'stop_memory')
            self.button('CANCEL / RESUME',470,457,340,'cancel_stop')

    def action(self,action):
        if action.startswith('analysis:'):
            self.analysis.action(action.split(':',1)[1]);self.layout();return
        try:
            if action in ('config','dashboard','analysis'):
                self.page=action
                if action=='analysis':self.analysis.refresh()
            elif action=='source':
                choices=['new','lab',*self.models.names(),'resume']
                self.source=choices[(choices.index(self.source)+1)%len(choices)]
            elif action=='model_configuration':
                if self.source!='new':raise ValueError('Saved models restore their own architecture and encoder. Choose new to select A/B/C/D.')
                keys=tuple(LINEUP)
                self.model_configuration=keys[(keys.index(self.model_configuration)+1)%4]
                self.text['hidden']=','.join(map(str,LINEUP[self.model_configuration][3]))
                self.text['activation']='relu'
            elif action=='select':self.selected+=1
            elif action=='advanced':self.advanced=not self.advanced
            elif action=='advanced_page':self.advanced_page=1-self.advanced_page
            elif action=='refresh':self.refresh=(25,100,500)[((25,100,500).index(self.refresh)+1)%3]
            elif action=='headless':self.headless=not self.headless
            elif action=='profiling':self.profiling=not self.profiling
            elif action=='environments':self.environments=(1,2,4,8)[((1,2,4,8).index(self.environments)+1)%4]
            elif action=='strategy':self.strategy=('auto','vectorized','multiprocessing')[(('auto','vectorized','multiprocessing').index(self.strategy)+1)%3]
            elif action=='update_policy':self.update_policy='vector' if self.update_policy=='transition' else 'transition'
            elif action=='start':
                config,architecture,execution=self.configuration()
                if self.worker and self.worker.thread.is_alive() and self.snapshot and self.snapshot.status=='Running':
                    raise ValueError('Pause or stop the current session before starting a new experiment')
                resume=network=None
                if self.source=='resume':
                    entries=self.entries()
                    if not entries:raise ValueError('No resumable checkpoints available')
                    resume=entries[self.selected%len(entries)]['path']
                    metadata=self.store.metadata(resume)
                    if metadata['kind']!='resumable-boundary':raise ValueError('Select a final or recovery checkpoint; model snapshots cannot resume')
                    config=replace(TrainingConfig.from_dict(metadata['hyperparameters']),episodes=config.episodes)
                    if config.episodes<metadata['episode']:raise ValueError('Target must include completed episodes')
                elif self.source=='lab':network=copy.deepcopy(self.lab_provider().network)
                elif self.source!='new':network,_=self.models.load(self.source)
                if self.worker:self.worker.shutdown()
                self.worker=FrontierWorker(self.profile,Path(self.text['output']))
                self.worker.start(config,execution,architecture,network,resume,self.text['name'].strip())
                self.page='dashboard';self.cache_key=None
            elif self.worker:
                if action=='files':
                    if not self.snapshot or not self.snapshot.experiment:raise ValueError('Start an experiment first')
                    path=Path(self.snapshot.experiment)
                    if sys.platform=='win32':os.startfile(path)
                    else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(path)])
                elif action=='stop':
                    if self.snapshot and self.snapshot.status=='Completed':
                        self.status='Training completed; final model and resumable checkpoint are already saved'
                        return
                    self.worker.command('pause');self.confirm=True
                elif action=='save_stop':
                    self.confirm=None;self.worker.command('save_stop')
                elif action=='stop_memory':
                    self.confirm=None;self.worker.command('stop')
                elif action=='cancel_stop':
                    self.confirm=None;self.worker.command('resume')
                elif action in ('pause','resume','save'):self.worker.command(action)
            self.layout()
        except (ValueError,TypeError,OSError,RuntimeError) as error:self.status=str(error)

    def checkpoint_path(self):return self.snapshot.checkpoint if self.snapshot else None

    def update(self,dt):
        self.analysis.update(dt,self.page=='analysis')
        if self.page=='analysis':
            self.buttons=[b for b in self.buttons if not b.action.startswith(self.analysis.prefix)]+self.analysis.buttons
        if self.worker:
            latest=self.worker.latest()
            if latest is not self.snapshot:
                self.snapshot=latest
                if latest:self.status=latest.message

    def handle_event(self,event,mouse):
        if self.confirm:return False
        return self.analysis.handle_event(event,mouse) if self.page=='analysis' else super().handle_event(event,mouse)

    def close(self):
        if self.worker:self.worker.shutdown()
        self.analysis.close()

    def dashboard(self,surface):
        snap=self.snapshot
        if not snap:
            label(surface,'Configure and start a Frontier experiment',(640,320),24,MUTED);return
        metrics=snap.metrics
        if not metrics:
            label(surface,snap.message[:120],(24,150),18,MUTED,False);return
        def number(key,formatting='.3g'):
            value=metrics.get(key)
            return 'N/A' if value is None else format(value,formatting)
        label(surface,f"{snap.status.upper()} / CPU / {metrics.get('environments',1)} ENVIRONMENTS / {metrics.get('strategy','vectorized').upper()} / UPDATES PER {metrics.get('update_policy','transition').upper()}",(24,122),17,GREEN,False)
        items=[f"Episodes {metrics['completed']:,} / {metrics['target']:,} ({metrics['progress']:.2f}%)",
            f"Mean100 {number('rolling_mean')} / median100 {number('rolling_median')} / best {number('best')}",
            f"Steps {metrics['steps']:,} / {number('steps_per_second')} steps/s / active {metrics.get('active_environments',0)} / {metrics.get('worker_health','Healthy')}",
            f"{number('episodes_per_second')} episodes/s / {number('optimizer_updates_per_second')} optimizer updates/s",
            f"Elapsed {duration(metrics.get('elapsed'))} / "+('Completed' if snap.status=='Completed' else 'ETA estimate: '+duration(metrics.get('eta'))),
            f"Epsilon {number('epsilon')} / Huber TD loss {number('loss')}",
            f"Apples {metrics['apples']:,} / optimizer updates {metrics['optimizer_updates']:,}",
            f"Collisions {metrics['collisions']:,} ({metrics['collision_frequency']*100:.2f}%) / stagnations {metrics['stagnations']:,} ({metrics['stagnation_frequency']*100:.2f}%)",
            f"Episode length (recent100): mean {number('mean_length')} / median {number('median_length')}",
            f"CPU {number('cpu_percent','.1f')}% (100%=one core) / RAM {metrics['ram_bytes']/1048576:.1f} MiB" if metrics['ram_bytes'] is not None else f"CPU {number('cpu_percent','.1f')}% / RAM N/A"]
        for index,text in enumerate(items):label(surface,text,(24+(index//5)*630,160+(index%5)*29),16,TEXT,False)
        charts=snap.charts or {}
        plot=AnalysisChart()
        for x,mode in ((24,'distribution'),(650,'mean_median')):
            plot.draw(surface,pygame.Rect(x,320,606,155),charts.get(mode),(-1,-1))
        chart(surface,pygame.Rect(24,490,606,145),'Huber loss / completed episodes (last 2,000)',snap.loss_trend)
        self.throughput_chart(surface,pygame.Rect(650,490,606,145),snap.throughput)

    def throughput_chart(self,surface,rect,points):
        pygame.draw.rect(surface,PANEL,rect,border_radius=6)
        label(surface,'Throughput / actual active seconds',(rect.x+12,rect.y+8),16,GREEN,False)
        if not points:return
        low,high=0,max(1,max(y for _,y in points))
        first,last=points[0][0],points[-1][0]
        coords=[(rect.x+15+(x-first)*(rect.width-30)/max(1e-9,last-first),rect.bottom-28-y*(rect.height-65)/high) for x,y in points]
        if len(coords)>1:pygame.draw.lines(surface,GREEN,False,coords,1)
        label(surface,f'{first:.3g}–{last:.3g} active s / 0–{high:.1f} steps/s',(rect.x+12,rect.bottom-22),12,MUTED,False)

    def draw(self,surface):
        surface.fill(BG)
        label(surface,'FRONTIER TRAINING',(24,15),27,GREEN,False)
        metrics=self.snapshot.metrics if self.snapshot and self.page!='config' else {}
        label(surface,f"CPU / {metrics.get('environments',self.environments)} environments / {metrics.get('strategy',self.strategy)}",(420,23),17,MUTED,False)
        if self.page=='analysis':
            window=pygame.display.get_surface()
            width,height=window.get_size()
            scale=min(width/1280,height/720)
            position=pygame.mouse.get_pos()
            mouse=((position[0]-(width-1280*scale)/2)/scale,(position[1]-(height-720*scale)/2)/scale)
            try:self.analysis.draw(surface,mouse)
            except Exception as error:label(surface,str(error)[:120],(640,360),16,MUTED)
            return
        if self.page=='config':
            if self.advanced:
                keys=self.advanced_keys()
                for index,key in enumerate(keys[self.advanced_page*12:(self.advanced_page+1)*12]):
                    label(surface,key.replace('_',' ').title(),(24+(index//6)*620,178+(index%6)*60),15,TEXT,False)
            else:
                try:
                    info=self.model_info()
                    if info:
                        shape='3x25x25 + heading4' if isinstance(info['input_shape'],dict) else str(info['input_shape'])
                        label(surface,f"{info['architecture_name']} / {info['encoder_name']} / {shape} / {info['parameter_count']:,} parameters",(24,163),13,GREEN,False)
                except (ValueError,TypeError,OSError):pass
                for title,x,y in (('MODEL NAME',24,185),('EPISODE BUDGET',24,260),('SEED',334,260),
                    ('RECOVERY INTERVAL (EP)',650,260),('ROTATING SLOTS',960,260),('EXPERIMENT OUTPUT DIRECTORY',24,442)):
                    label(surface,title,(x,y),14,MUTED,False)
                label(surface,'A: Legacy11 / B: Distance19 / C: Binary CNN / D: Ordered CNN. One shared learner and replay.',(24,572),15,TEXT,False)
                label(surface,'Vector-step updates use fewer updates per transition. Resume restores saved collection and learning settings.',(24,602),14,MUTED,False)
            self.draw_fields(surface)
        else:
            if self.cache is None or self.cache_key is not self.snapshot:
                self.cache=pygame.Surface((1280,720));self.cache.fill(BG)
                self.dashboard(self.cache);self.cache_key=self.snapshot
            if self.cache is not None:surface.blit(self.cache,(0,115),pygame.Rect(0,115,1280,525))
            if self.snapshot and self.snapshot.model:
                label(surface,self.snapshot.model,(24,636),10,MUTED,False)
        label(surface,self.status[:170],(24,699),12,MUTED,False)
        if self.confirm:
            pygame.draw.rect(surface,PANEL,(210,305,830,210),border_radius=10)
            label(surface,'Preserve the current training state?',(640,335),24,GREEN)
            label(surface,'Finish the current episode before saving a resumable boundary checkpoint.',(640,375),16,TEXT)
