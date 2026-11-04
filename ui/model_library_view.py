"""Shared family → generation → checkpoint picker and reversible management screen."""
from pathlib import Path
import os
import sys
import subprocess
import pygame
from ai.model_library import FAMILIES
from ai.model_registry import identity,parse_model_config
from ui.ai_components import FormView
from ui.components import BG,PANEL,GREEN,TEXT,MUTED,label


class ModelLibraryView(FormView):
    def __init__(self,library):
        super().__init__('library:')
        self.library=library;self.target='arena';self.caller='ai'
        self.family='Alpha';self.generation=None;self.checkpoint=None
        self.expanded=set(FAMILIES);self.page=0;self.cp_page=0
        self.text={'search':'','description':''};self.confirm=None
        self.trash_mode=False;self.trash_selected=None
        self.review_mode=False
        self.preview_page=0
        self.details_mode=False;self.details_page=0
        self.summary={};self.refresh()

    def open(self,target,caller):
        self.target,self.caller=target,caller;self.confirm=None;self.trash_mode=False;self.details_mode=False;self.review_mode=False
        self.refresh()

    def refresh(self):
        self.summary=self.library.scan()
        self.generations=self.library.generations()
        self.status=f"Indexed {len(self.generations)} generations / {self.summary['checkpoints_added']} new checkpoints / {len(self.summary['review'])} items for review"
        self.layout()

    def tree(self):
        rows=[];query=self.text['search'].strip().casefold()
        for family in FAMILIES:
            generations=[g for g in self.generations if g['family']==family and (not query or query in (g['display_name']+' '+str(g['number'])).casefold())]
            if query and not generations and query not in family.casefold():continue
            rows.append(('family',family,('− ' if family in self.expanded else '+ ')+family))
            if family in self.expanded:
                rows.extend(('generation',g['id'],f"   Gen {g['number']} / {g['status']}"+(' | '+g['description'] if g['description'] else '')) for g in generations)
        return rows

    def layout(self):
        self.buttons=[];self.fields={}
        if self.confirm:
            self.button('MOVE TO TRASH',400,535,230,'confirm',True);self.button('CANCEL',650,535,230,'cancel')
            self.button('< FILES',220,490,140,'preview_previous');self.button('FILES >',880,490,140,'preview_next')
            return
        self.button('BACK',1080,24,176,'back')
        self.button('MIGRATION REVIEW',940,72,292,'review')
        if self.details_mode:
            self.button('RETURN TO CHECKPOINTS',24,638,340,'details')
            self.button('MORE DETAILS',390,638,300,'details_next')
            self.button('OPEN FILES',720,638,300,'files')
            return
        self.field('search',24,110,430)
        if self.review_mode:
            self.button('REFRESH INDEX',480,590,350,'refresh')
            self.button('RETURN TO LIBRARY',850,590,350,'review')
        elif self.trash_mode:
            entries=self.library.trashed()
            for i,item in enumerate(entries[self.page*12:(self.page+1)*12]):
                self.button(item['kind'].upper()+' / '+item['target'][:12],24,165+i*32,430,'trash_item:'+item['id'],item['id']==self.trash_selected)
                self.buttons[-1].rect.height=28
            self.button('RESTORE SELECTED',480,590,340,'restore',True)
            self.button('BACK TO LIBRARY',850,590,350,'trash')
        else:
            for i,(kind,identifier,title) in enumerate(self.tree()[self.page*12:(self.page+1)*12]):
                self.button(title[:55],24,165+i*30,430,kind+':'+identifier,identifier in (self.generation,self.family))
                self.buttons[-1].rect.height=27
            if self.generation:
                cps=self.library.checkpoints(self.generation)
                for i,cp in enumerate(cps[self.cp_page*6:(self.cp_page+1)*6]):
                    title=f"ep {cp['episode'] if cp['episode'] is not None else '?'} / {cp['kind'].upper()} / {cp['status']} / {cp['id'][:8]}"
                    self.button(title,480,285+i*31,752,'checkpoint:'+cp['id'],cp['id']==self.checkpoint)
                    self.buttons[-1].rect.height=28
                self.button('< CHECKPOINTS',480,475,160,'cp_previous');self.button('CHECKPOINTS >',650,475,160,'cp_next')
                self.button('DETAILS',820,475,170,'details');self.button('OPEN FILES',1000,475,232,'files')
            self.field('description',480,535,752)
            self.button('NEW '+self.family.upper()+' GENERATION',24,590,285,'new',True)
            seed=self.checkpoint and self.target in ('frontier_resume','legacy_resume')
            self.button('SEED NEW EXPERIMENT' if seed else 'CONFIGURE CREATED',324,590,275,'seed' if seed else 'configure')
            self.button('LOAD SELECTED',614,590,310,'load',True)
            self.button('SAVE DESCRIPTION',940,590,292,'describe')
            if self.target=='analysis_b':
                self.buttons[-1].text='NONE — SINGLE MODEL';self.buttons[-1].action='library:none'
            self.button('DELETE CHECKPOINT',24,638,285,'delete_checkpoint')
            self.button('DELETE GENERATION',324,638,275,'delete_generation')
            self.button('TRASH / RESTORE',614,638,310,'trash')
            self.button('UNLOAD IDLE VIEWS',940,638,292,'unload')
        self.button('< GENERATIONS',24,548,200,'previous');self.button('GENERATIONS >',234,548,220,'next')

    def action(self,command):
        try:
            if command.startswith('family:'):
                family=command.split(':',1)[1];self.family=family
                if family in self.expanded:self.expanded.remove(family)
                else:self.expanded.add(family)
            elif command.startswith('generation:'):
                self.generation=command.split(':',1)[1];g=self.library.get(self.generation)
                self.family=g['family'];self.checkpoint=None;self.cp_page=0;self.text['description']=g['description']
            elif command.startswith('checkpoint:'):self.checkpoint=command.split(':',1)[1]
            elif command in ('previous','next'):self.page=max(0,self.page+(-1 if command=='previous' else 1))
            elif command in ('cp_previous','cp_next'):self.cp_page=max(0,self.cp_page+(-1 if command=='cp_previous' else 1))
            elif command=='describe':
                if not self.generation:raise ValueError('Select a generation first')
                self.library.describe(self.generation,self.text['description']);self.refresh()
            elif command.startswith('delete_'):
                kind='checkpoint' if command=='delete_checkpoint' else 'generation'
                identifier=self.checkpoint if kind=='checkpoint' else self.generation
                if not identifier:raise ValueError('Select the exact item to delete')
                self.confirm=self.library.preview_delete(kind,identifier)
                self.preview_page=0
            elif command in ('preview_previous','preview_next'):
                self.preview_page=max(0,self.preview_page+(-1 if command=='preview_previous' else 1))
            elif command=='cancel':self.confirm=None
            elif command=='confirm':
                self.library.delete(self.confirm);self.confirm=None;self.checkpoint=None
                if self.generation and self.library.get(self.generation)['status']=='Deleted':self.generation=None
                self.refresh();self.status='Moved to application Trash. Restore is available; no permanent deletion performed.'
            elif command=='trash':self.trash_mode=not self.trash_mode;self.page=0
            elif command=='review':self.review_mode=not self.review_mode
            elif command=='details':self.details_mode=not self.details_mode;self.details_page=0
            elif command=='details_next':self.details_page+=1
            elif command=='refresh':self.refresh()
            elif command.startswith('trash_item:'):self.trash_selected=command.split(':',1)[1]
            elif command=='restore':
                if not self.trash_selected:raise ValueError('Select an item in Trash')
                self.library.restore(self.trash_selected);self.refresh();self.status='Restored original identity and files'
            elif command=='files':
                if not self.checkpoint:raise ValueError('Select a checkpoint first')
                path=Path(self.library.checkpoint(self.checkpoint)['path']);folder=path if path.is_dir() else path.parent
                if not folder.exists():raise ValueError('Storage is missing')
                if sys.platform=='win32':os.startfile(folder)
                else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(folder)])
            self.layout()
        except (ValueError,OSError,RuntimeError) as error:self.status=str(error);self.layout()

    def handle_event(self,event,mouse):
        if self.confirm:
            if event.type==pygame.MOUSEWHEEL:
                self.action('preview_previous' if event.y>0 else 'preview_next');return True
            return False
        result=super().handle_event(event,mouse)
        if result and self.focus=='search':self.page=0;self.layout()
        if event.type==pygame.MOUSEWHEEL:
            self.action('previous' if event.y>0 else 'next');return True
        return result

    def draw(self,surface):
        surface.fill(BG)
        label(surface,'MODEL LIBRARY',(24,22),28,GREEN,False)
        label(surface,'FAMILY → GENERATION → CHECKPOINT / '+self.target.upper(),(400,33),15,MUTED,False)
        if not self.details_mode:label(surface,'SEARCH FAMILY / GENERATION / DESCRIPTION',(24,85),13,MUTED,False)
        if self.details_mode:
            g=self.library.get(self.generation)
            cp=self.library.checkpoint(self.checkpoint) if self.checkpoint else None
            info={**identity(parse_model_config(cp['data'])),**cp['data']} if cp else {}
            settings=info.get('hyperparameters') or g['data'].get('training') or {}
            items=[('Name',g['name']),('Description',g['description']),('Created UTC',g['created']),
                ('Generation ID',g['id']),('Status',g['status']),('Training interfaces',', '.join(g['data'].get('interfaces',[])) or 'Unknown'),
                ('Checkpoint ID',cp['id'] if cp else 'None'),('Storage',cp['path'] if cp else 'No checkpoint'),
                ('Architecture',info.get('architecture_name','Unknown')),('Encoder',info.get('encoder_name','Unknown'))]
            items.extend((key.replace('_',' '),value) for key,value in settings.items())
            items.extend((reference['kind'],reference['path']) for reference in self.library.references(g['id']))
            pages=max(1,(len(items)+31)//32);self.details_page%=pages
            for i,(key,value) in enumerate(items[self.details_page*32:(self.details_page+1)*32]):
                label(surface,f'{key}: {value}'[:92],(24+(i//16)*620,135+(i%16)*28),14,TEXT,False)
            label(surface,f'Details page {self.details_page+1}/{pages}',(24,605),15,MUTED,False)
        elif self.review_mode:
            label(surface,'MIGRATION SUMMARY / ITEMS REQUIRING REVIEW',(24,165),22,GREEN,False)
            items=self.summary.get('review',[])
            if not items:label(surface,'No unresolved metadata or lineage was discovered.',(24,210),17,TEXT,False)
            for i,item in enumerate(items[self.page*12:(self.page+1)*12]):label(surface,str(item)[-145:],(24,210+i*25),13,TEXT,False)
        elif self.trash_mode:
            label(surface,'REVERSIBLE TRASH',(480,150),24,GREEN,False)
            label(surface,'Restore preserves generation numbers and immutable identities.',(480,190),16,TEXT,False)
            label(surface,'Restore refuses to overwrite existing files.',(480,220),16,MUTED,False)
        elif self.generation:
            g=self.library.get(self.generation)
            cps=self.library.checkpoints(self.generation)
            cp=next((c for c in cps if c['id']==self.checkpoint),None)
            info={**identity(parse_model_config(cp['data'])),**cp['data']} if cp else {}
            lines=[g['name'],g['status']+' / ID '+g['id'],
                info.get('architecture_name','Select a checkpoint to inspect architecture'),
                f"Episode {info.get('episode',g['data'].get('episode','Unknown'))} / steps {info.get('environment_steps',g['data'].get('environment_steps','Unknown'))}",
                f"Seed {info.get('hyperparameters',g['data'].get('training') or {}).get('seed','Unknown')} / evaluations {sum(r['kind'].startswith('evaluation') for r in self.library.references(g['id']))}",
                (cp or {}).get('path','No checkpoint selected; an experiment record is not a loadable model')]
            for i,line in enumerate(lines):label(surface,str(line)[:105],(480,115+i*26),18 if i==0 else 14,GREEN if i==0 else TEXT,False)
        else:
            label(surface,'Choose a generation, then an exact checkpoint.',(480,165),21,GREEN,False)
            label(surface,'Frontier is a training interface. Families identify the network.',(480,205),16,MUTED,False)
            label(surface,'No checkpoints or evaluations are invented.',(480,238),16,MUTED,False)
        if not (self.details_mode or self.review_mode or self.trash_mode):label(surface,'OPTIONAL EXPERIMENT DESCRIPTION',(480,514),13,MUTED,False)
        self.draw_fields(surface)
        label(surface,self.status[:165],(24,697),13,MUTED,False)
        if self.confirm:
            overlay=pygame.Surface((1280,720),pygame.SRCALPHA);overlay.fill((0,0,0,210));surface.blit(overlay,(0,0))
            pygame.draw.rect(surface,PANEL,(190,145,900,440),border_radius=10)
            label(surface,'CONFIRM REVERSIBLE DELETION',(220,170),24,GREEN,False)
            plan=self.confirm
            label(surface,f"{plan['kind'].upper()} / {len(plan['files'])} files / {len(plan['retained_shared'])} shared records retained",(220,210),16,TEXT,False)
            items=plan['files'] or plan['paths']
            for i,path in enumerate(items[self.preview_page*10:(self.preview_page+1)*10]):label(surface,path[-110:],(220,246+i*23),13,TEXT,False)
            label(surface,f'File preview page {self.preview_page+1} / reversible move; shared records remain',(390,496),13,MUTED,False)
