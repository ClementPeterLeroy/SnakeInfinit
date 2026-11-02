"""Training configuration/dashboard; worker tensors never enter drawing code."""
import copy
import time
from dataclasses import replace
from pathlib import Path
import pygame
from ai.config import NetworkConfig, ACTIVATIONS
from ai.training_config import TrainingConfig
from ai.checkpoints import CheckpointStore
from ai.persistence import ModelStore
from ai.worker import TrainingWorker
from ui.ai_components import FormView, chart, draw_board
from ui.components import BG,TEXT,GREEN,MUTED,label
from ui.analysis_view import AnalysisView


class TrainingView(FormView):
    VIEWS=(('realtime',1),('accelerated',2),('accelerated',5),('accelerated',10),('accelerated',20),('headless',1))
    def __init__(self,root,lab_provider):
        super().__init__('train:')
        self.store=CheckpointStore(Path(root)/'checkpoints')
        self.models=ModelStore(Path(root)/'models')
        self.lab_provider=lab_provider
        self.worker=None
        self.snapshot=None
        self.page='config'
        self.config_page=0
        self.view_index=0
        self.source='new'
        self.selected=0
        self.confirm=None
        self.count_key=None
        self.actual_count=0
        self.rolling=()
        self.library_time=0.
        self.library=[]
        self.text={key:str(value) for key,value in TrainingConfig().to_dict().items()}
        self.text.update(hidden='16,8',activation='relu',name='Snake DQN Alpha')
        self.status='Configure a new run or select a trusted local checkpoint to resume'
        self.error=None
        self.analysis=AnalysisView(root)
        self.rolling_median=()
        self.layout()

    def entries(self):
        now=time.monotonic()
        if now-self.library_time>1:
            self.library=self.store.entries()
            self.library_time=now
        return self.library

    def layout(self):
        self.buttons,self.fields=[],{}
        self.button('CONFIGURATION',24,67,215,'config',self.page=='config')
        self.button('DASHBOARD',251,67,215,'dashboard',self.page=='dashboard')
        self.button('ANALYSIS',478,67,215,'analysis',self.page=='analysis')
        self.button('BACK TO AI',1090,67,166,'back')
        if self.confirm:
            self.button('CONFIRM NEW RUN',430,370,205,'confirm',True)
            self.button('CANCEL',650,370,160,'cancel')
            return
        if self.page=='analysis':
            self.analysis.layout()
            self.buttons.extend(self.analysis.buttons)
        elif self.page=='config':
            self.button(f'SOURCE: {self.source}',24,120,570,'source')
            self.button('NEXT SETTINGS PAGE',930,120,326,'page')
            keys=list(TrainingConfig().to_dict())+['hidden','activation','name']
            for position,key in enumerate(keys[self.config_page*12:(self.config_page+1)*12]):
                column,row=position//6,position%6
                self.field(key,24+column*620,204+row*64,570)
            self.button('START TRAINING',24,615,270,'start',True)
            self.button('SELECT SAVED CHECKPOINT',310,615,310,'select')
            self.button('RESUME SELECTED',636,615,280,'resume_saved')
        else:
            for i,(title,action) in enumerate((('PAUSE / RESUME','pause'),('STOP','stop'),('SAVE CHECKPOINT','save'),
                                               ('VIEW SPEED','view'),('OPEN IN ARENA','arena'),('INSPECT IN LAB','lab'))):
                self.button(title,24+i*207,642,195,action,i==0)

    def configuration(self):
        defaults=TrainingConfig().to_dict()
        values={key:int(self.text[key]) if type(default) is int else float(self.text[key]) for key,default in defaults.items()}
        config=TrainingConfig(**values)
        architecture=NetworkConfig(tuple(int(v.strip()) for v in self.text['hidden'].split(',')),self.text['activation'],config.seed)
        if not self.text['name'].strip() or len(self.text['name'])>40:
            raise ValueError('Model name must contain 1..40 characters')
        return config,architecture

    def start(self,resume=None):
        config,architecture=self.configuration()
        network=None
        if not resume:
            if self.source=='lab':
                network=copy.deepcopy(self.lab_provider().network)
            elif self.source!='new':
                network,_=self.models.load(self.source)
        if self.worker:
            self.worker.shutdown()
        self.worker=TrainingWorker(self.store)
        self.view_index=0
        self.worker.start(config,architecture,network,resume,self.text['name'].strip(),config.episodes if resume else None)
        self.page='dashboard'
        self.status='Loading trusted resume state' if resume else 'Starting CPU DQN'

    def action(self,action):
        self.error=None
        if action.startswith('analysis:'):
            self.analysis.action(action.split(':',1)[1])
            self.layout()
            return
        try:
            if action=='cancel':
                self.confirm=None
            elif action=='confirm':
                resume=self.confirm[1]
                self.confirm=None
                self.start(resume)
            elif action in ('config','dashboard','analysis'):
                self.page=action
                if action=='analysis':self.analysis.refresh()
            elif action=='page':
                self.config_page=1-self.config_page
            elif action=='source':
                sources=['new','lab',*self.models.names()]
                self.source=sources[(sources.index(self.source)+1)%len(sources)] if self.source in sources else 'new'
            elif action=='start':
                self.configuration()
                if self.worker:
                    self.confirm=('start',None)
                else:
                    self.start()
            elif action=='select':
                self.selected+=1
            elif action=='resume_saved':
                entries=self.entries()
                if not entries:
                    raise ValueError('No checkpoints; start training to create a model library')
                path=entries[self.selected%len(entries)]['path']
                if self.worker:
                    self.confirm=('resume',path)
                else:
                    self.start(path)
            elif self.worker:
                if action=='pause':
                    self.worker.command('pause' if self.snapshot and self.snapshot.status=='Running' else 'resume')
                elif action=='stop':
                    self.worker.command('stop')
                elif action=='save':
                    self.worker.command('save')
                elif action=='view':
                    self.view_index=(self.view_index+1)%len(self.VIEWS)
                    self.worker.command('view',self.VIEWS[self.view_index])
        except (ValueError,OSError,RuntimeError) as error:
            self.status=str(error)
            self.error=str(error)
        self.layout()

    def update(self,dt):
        self.analysis.update(dt,self.page=='analysis')
        if self.page=='analysis' and not self.confirm:
            # Background catalog/statistics callbacks can replace Analysis controls.
            self.buttons=[button for button in self.buttons if not button.action.startswith(self.analysis.prefix)]+self.analysis.buttons
        if self.worker:
            snapshot=self.worker.latest()
            if snapshot is not None:
                if snapshot is not self.snapshot:
                    scores=dict(snapshot.histories).get('scores',())
                    running=0.
                    rolling=[]
                    for i,value in enumerate(scores):
                        running+=value
                        if i>=100:
                            running-=scores[i-100]
                        rolling.append(running/min(i+1,100))
                    self.rolling=tuple(rolling)
                    from statistics import median
                    self.rolling_median=tuple(median(scores[max(0,i-99):i+1]) for i in range(max(0,len(scores)-100),len(scores)))
                self.snapshot=snapshot
                self.status=snapshot.message

    def checkpoint_path(self):
        return self.snapshot.checkpoint if self.snapshot else None

    def close(self):
        self.analysis.close()
        if self.worker:
            self.worker.shutdown()

    def handle_event(self,event,mouse):
        if self.page=='analysis':
            return self.analysis.handle_event(event,mouse)
        return super().handle_event(event,mouse)

    def draw(self,surface):
        surface.fill(BG)
        label(surface,'TRAINING CENTER',(24,15),27,GREEN,False)
        label(surface,'Train & monitor',(330,23),17,MUTED,False)
        if self.page=='analysis':
            # Mouse coordinates use the same logical viewport as all other controls.
            window=pygame.display.get_surface()
            width,height=window.get_size()
            scale=min(width/1280,height/720)
            position=pygame.mouse.get_pos()
            mouse=((position[0]-(width-1280*scale)/2)/scale,(position[1]-(height-720*scale)/2)/scale)
            try:
                self.analysis.draw(surface,mouse)
            except Exception as error:
                self.analysis.status='Analysis display error: '+str(error)
                label(surface,self.analysis.status[:120],(640,340),16,MUTED)
            return
        elif self.page=='config':
            keys=list(TrainingConfig().to_dict())+['hidden','activation','name']
            for position,key in enumerate(keys[self.config_page*12:(self.config_page+1)*12]):
                column,row=position//6,position%6
                label(surface,key.replace('_',' ').title(),(24+column*620,180+row*64),16,TEXT,False)
            try:
                _,arch=self.configuration()
                key=(self.source,arch)
                if key!=self.count_key:
                    from ai.network import SnakeNetwork
                    model=self.lab_provider().network if self.source=='lab' else self.models.load(self.source)[0] if self.source!='new' else SnakeNetwork(arch)
                    self.actual_count=model.parameter_count
                    self.count_key=key
                label(surface,f'{self.actual_count:,} trainable parameters / page {self.config_page+1}/2',(24,163),13,GREEN,False)
            except (ValueError,OSError):
                pass
            entries=self.entries()
            if entries:
                selected=entries[self.selected%len(entries)]
                label(surface,f"Trusted local selection: {selected['name']} / episode {selected['episode']}",(24,575),16,MUTED,False)
        elif self.snapshot:
            snap=self.snapshot
            m=snap.values
            if not m:
                label(surface,snap.status+' / '+snap.message[:100],(24,190),20,TEXT,False)
                label(surface,(self.error or self.status)[:150],(24,696),13,MUTED,False)
                return
            label(surface,snap.status.upper(),(24,120),24,GREEN,False)
            mode='HEADLESS / no per-step rendering' if snap.mode=='headless' else f'{snap.mode.upper()} / target {snap.multiplier}x'
            label(surface,mode,(24,157),15,MUTED,False)
            if snap.mode!='headless':
                draw_board(surface,snap.state,pygame.Rect(24,192,280,280))
            lines=[f"Episode {m['episode']} / completed {m['completed']}",f"Score {m['score']} / best {m['best']} / avg100 {m['average']:.2f}",
                f"Steps {m['steps']} / apples {m['apples']}",f"Epsilon {m['epsilon']:.5f} / loss {m['loss'] if m['loss'] is not None else '--'}",
                f"Collisions {m['collisions']} / stagnations {m['stagnations']}",
                f"Elapsed {m['elapsed']:.1f}s / CPU / parameters {m['parameter_count']}",f"Achieved {snap.steps_per_second:.1f} steps/s / {snap.episodes_per_second:.2f} eps/s",
                f"Avg length {m['average_length']:.1f} / reward total {m['reward_total']:.2f}",
                f"Gradient norm {m['gradient_norm'] if m['gradient_norm'] is not None else '--'}",
                f"Checkpoint {'PENDING' if snap.pending_save else 'saved' if snap.checkpoint else '--'}"]
            for index,line in enumerate(lines):
                label(surface,line[:72],(330,123+index*24),14,TEXT,False)
            histories=dict(snap.histories)
            scores=histories.get('scores',())
            chart(surface,pygame.Rect(770,120,486,230),'Episode scores',scores)
            chart(surface,pygame.Rect(770,370,486,230),'Rolling average (100)',self.rolling)
            chart(surface,pygame.Rect(24,490,350,130),'Huber TD loss',histories.get('losses',()))
            pygame.draw.rect(surface,(26,32,31),(394,390,350,230),border_radius=6)
            label(surface,f"EXPLORATION ε: {m['epsilon']:.5f}",(410,405),18,GREEN,False)
            recent=scores[-100:]
            label(surface,f'Recent episodes: {len(recent)} / best {m["best"]}',(410,447),17,TEXT,False)
            label(surface,f'Rolling mean: {self.rolling[-1]:.3f}' if self.rolling else 'Rolling mean: --',(410,483),17,TEXT,False)
            label(surface,f'Rolling median: {self.rolling_median[-1]:.3f}' if self.rolling_median else 'Rolling median: --',(410,519),17,TEXT,False)
            label(surface,'Detailed distributions and exports: Analysis',(410,566),13,MUTED,False)
        else:
            label(surface,'Start training or resume a checkpoint from Configuration',(640,320),22,MUTED)
        self.draw_fields(surface)
        label(surface,(self.error or self.status)[:150],(24,696),13,MUTED,False)
        if self.confirm:
            pygame.draw.rect(surface,BG,(300,260,680,180),border_radius=10)
            label(surface,'Replace the active run? Save a checkpoint first if needed.',(640,310),20,TEXT)
