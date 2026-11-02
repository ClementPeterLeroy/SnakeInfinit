"""Board-first spectator interface and grouped model/checkpoint selection."""
from pathlib import Path
import math
import time
import pygame
from ai.arena import Arena
from ai.checkpoints import CheckpointStore
from ai.training_config import TrainingConfig
from ai.model_registry import identity,parse_model_config
from ui.ai_components import FormView, draw_board
from ui.components import BG,GREEN,TEXT,MUTED,label


class ArenaView(FormView):
    def __init__(self,root):
        super().__init__('arena:')
        self.store=CheckpointStore(Path(root)/'checkpoints')
        self.arena=None
        self.page='select'
        self.run_index=self.checkpoint_index=0
        self.text={'seed':'0','delay':'1'}
        self.status='Select a trusted local model and checkpoint'
        self.selected_metadata=None
        self.library_time=0.
        self.library=[]
        self.metadata_key=None
        self.selection_metadata=None
        self.displayed_seed=0
        self.layout()

    def selection(self):
        now=time.monotonic()
        if now-self.library_time>1:
            self.library=self.store.entries()
            self.library_time=now
        entries=self.library
        runs=sorted({e['run'] for e in entries})
        if not runs:
            return None
        selected=[e for e in entries if e['run']==runs[self.run_index%len(runs)]]
        return selected[self.checkpoint_index%len(selected)]

    def layout(self):
        self.buttons,self.fields=[],{}
        self.button('BACK TO AI',1090,65,166,'back')
        if self.page=='select':
            entry=self.selection()
            self.button('MODEL: '+(entry['name']+' / run '+entry['run'][:8] if entry else 'No saved training runs'),80,180,960,'model')
            self.button('CHECKPOINT: '+(str(entry['episode'])+' episodes' if entry else '--'),80,245,960,'checkpoint')
            self.button('LOAD TRUSTED CHECKPOINT',80,445,400,'load',True)
            self.field('seed',710,447,330)
        elif self.arena:
            arena=self.arena
            self.field('seed',1000,449,240)
            self.field('delay',1110,491,130)
            for index,(title,action) in enumerate((('PAUSE' if arena.running else 'PLAY / RESUME','play'),
                ('RESET SAME SEED','reset'),('NEW GAME','new'),('NEXT GAME','next'),
                (f'SPEED {arena.multiplier}x','speed'),('AUTO NEXT: '+('ON' if arena.auto_next else 'OFF'),'auto'),
                ('SELECT MODEL','select'))):
                self.button(title,800+(index%2)*225,545+(index//2)*37,215,action,index==0)

    def load(self,path):
        network,metadata=self.store.load_model(path)
        seed=int(self.text['seed'])
        if not 0<=seed<2**63:
            raise ValueError('Seed must be an integer from 0 to 2^63-1')
        self.arena=Arena(network,TrainingConfig.from_dict(metadata['hyperparameters']),seed,
                         metadata['model_name'],metadata['episode'])
        self.selected_metadata=metadata
        self.displayed_seed=seed
        self.page='watch'
        self.status='Frozen greedy evaluation / no training or epsilon exploration'
        self.layout()

    def action(self,action):
        try:
            if action=='model':
                self.run_index+=1
                self.checkpoint_index=0
            elif action=='checkpoint':
                self.checkpoint_index+=1
            elif action=='load':
                entry=self.selection()
                if not entry:
                    raise ValueError('No saved checkpoints; create a training run first')
                self.load(entry['path'])
            elif self.arena:
                arena=self.arena
                if action=='play':
                    arena.pause() if arena.running else arena.play()
                elif action=='reset':
                    arena.reset(int(self.text['seed']))
                elif action in ('new','next'):
                    if action=='next' and not arena.environment.done:
                        raise ValueError('Next Game is available after the current game ends')
                    arena.new_game()
                    self.text['seed']=str(arena.seed)
                    arena.play()
                elif action=='speed':
                    arena.multiplier=(1,2,5,10)[((1,2,5,10).index(arena.multiplier)+1)%4]
                    arena.accumulator=0.
                elif action=='auto':
                    delay=float(self.text['delay'])
                    if not math.isfinite(delay) or not 0<=delay<=60:
                        raise ValueError('Auto Next delay must be 0..60 seconds')
                    arena.auto_delay=delay
                    arena.auto_next=not arena.auto_next
                    if not arena.auto_next and arena.environment.done:
                        arena.pause()
                elif action=='select':
                    arena.pause()
                    self.page='select'
        except (ValueError,OSError,RuntimeError) as error:
            self.status=str(error)
        self.layout()

    def update(self,dt):
        if self.page=='watch' and self.arena:
            try:
                self.arena.advance(dt)
                if self.arena.seed!=self.displayed_seed:
                    self.text['seed']=str(self.arena.seed)
                    self.displayed_seed=self.arena.seed
            except (ValueError,RuntimeError) as error:
                self.status=str(error)
            self.layout()

    def pause(self):
        if self.arena:
            self.arena.pause()
        self.layout()

    def draw(self,surface):
        surface.fill(BG)
        label(surface,'ARENA',(24,18),30,GREEN,False)
        label(surface,'Watch & evaluate',(180,28),18,MUTED,False)
        if self.page=='select':
            entry=self.selection()
            label(surface,'Choose a model, then a saved episode checkpoint',(80,125),24,TEXT,False)
            if entry:
                try:
                    if self.metadata_key!=entry['path']:
                        self.selection_metadata=self.store.metadata(entry['path'])
                        self.metadata_key=entry['path']
                    metadata=self.selection_metadata
                    info=identity(parse_model_config(metadata))
                    label(surface,info['architecture_name']+' / '+info['encoder_name'],(80,330),20,TEXT,False)
                    label(surface,f"Training episode {metadata['episode']} / environment steps {metadata['environment_steps']}",(80,375),20,MUTED,False)
                except (ValueError,OSError,KeyError) as error:
                    self.status=str(error)
            label(surface,'Environment seed:',(710,417),18,MUTED,False)
            label(surface,'Checkpoint files are loaded only from your local model library.',(80,520),18,MUTED,False)
        elif self.arena:
            arena=self.arena
            state=arena.environment.game.get_state()
            draw_board(surface,state,pygame.Rect(70,100,560,560))
            lines=[arena.name,f'Checkpoint episode {arena.checkpoint_episode}',
                f'Score {state.score} / steps {arena.environment.steps}',f'Simulation time {state.elapsed:.1f}s / {state.direction.name}',
                f'Session best {arena.best} / games {arena.games}',f'Average score {arena.total_score/max(1,arena.games):.2f}',
                'Last termination: '+str(arena.last_reason or '--'),
                'GAME OVER: '+str(arena.environment.reason) if arena.environment.done else ('PLAYING' if arena.running else 'PAUSED'),
                'Greedy / frozen weights / CPU']
            info=identity(arena.network.config)
            label(surface,info['architecture_name']+' / '+info['encoder_name'],(800,420),13,MUTED,False)
            for index,line in enumerate(lines):
                label(surface,line[:55],(800,130+index*32),19,GREEN if index in (0,7) else TEXT,False)
            label(surface,'Reset seed:',(800,455),17,MUTED,False)
            label(surface,'Auto-next delay (seconds):',(800,497),17,MUTED,False)
        self.draw_fields(surface)
        label(surface,self.status[:150],(24,697),13,MUTED,False)
