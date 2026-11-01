"""Small editable forms, actual-data charts and reusable Classic board rendering."""
import pygame
from ui.components import Button, BG, PANEL, BORDER, GREEN, MUTED, TEXT, label


class FormView:
    def __init__(self,prefix):
        self.prefix=prefix
        self.buttons=[]
        self.fields={}
        self.text={}
        self.focus=None
        self.status=''

    def button(self,title,x,y,width,action,primary=False):
        self.buttons.append(Button(title,pygame.Rect(x,y,width,36),self.prefix+action,primary=primary,text_size=15))

    def field(self,key,x,y,width):
        self.fields[key]=pygame.Rect(x,y,width,32)

    def handle_event(self,event,mouse):
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            self.focus=next((key for key,rect in self.fields.items() if rect.collidepoint(mouse)),None)
            if self.focus:
                pygame.key.start_text_input()
                return True
            pygame.key.stop_text_input()
        if self.focus and event.type==pygame.TEXTINPUT:
            self.text[self.focus]=(self.text[self.focus]+''.join(c for c in event.text if c.isprintable()))[:80]
            return True
        if self.focus and event.type==pygame.KEYDOWN:
            if event.key==pygame.K_BACKSPACE:
                self.text[self.focus]=self.text[self.focus][:-1]
            elif event.key in (pygame.K_RETURN,pygame.K_ESCAPE):
                self.focus=None
                pygame.key.stop_text_input()
            elif event.key==pygame.K_a and getattr(event,'mod',0)&pygame.KMOD_CTRL:
                self.text[self.focus]=''
            return True
        return False

    def draw_fields(self,surface):
        for key,rect in self.fields.items():
            pygame.draw.rect(surface,PANEL,rect,border_radius=5)
            pygame.draw.rect(surface,GREEN if self.focus==key else BORDER,rect,1,border_radius=5)
            old=surface.get_clip()
            surface.set_clip(rect.inflate(-8,-2))
            label(surface,self.text[key],(rect.x+6,rect.y+5),16,TEXT,False)
            surface.set_clip(old)


def draw_board(surface,state,rect):
    pygame.draw.rect(surface,PANEL,rect,border_radius=8)
    if state is None:
        return
    cell=min(rect.width/state.board_width,rect.height/state.board_height)
    for index,(x,y) in enumerate(state.snake):
        segment=pygame.Rect(rect.x+x*cell+1,rect.y+y*cell+1,max(1,cell-1),max(1,cell-1))
        pygame.draw.rect(surface,GREEN if index==0 else (58,143,92),segment,border_radius=2)
    if state.apple:
        x,y=state.apple
        pygame.draw.circle(surface,(229,124,119),(round(rect.x+(x+.5)*cell),round(rect.y+(y+.5)*cell)),max(2,round(cell*.35)))


def chart(surface,rect,title,values):
    pygame.draw.rect(surface,PANEL,rect,border_radius=6)
    label(surface,title,(rect.x+12,rect.y+8),16,GREEN,False)
    if not values:
        label(surface,'Waiting for actual training data',rect.center,14,MUTED)
        return
    # Keep at most ~320 points, retaining each bin's min/max rather than hiding spikes.
    stride=max(1,(len(values)+159)//160)
    points=[]
    for index in range(0,len(values),stride):
        batch=values[index:index+stride]
        points.extend((min(batch),max(batch)))
    low,high=min(points),max(points)
    span=high-low or 1.
    coordinates=[(rect.x+14+i*(rect.width-28)/max(1,len(points)-1),
                  rect.bottom-25-(value-low)*(rect.height-62)/span) for i,value in enumerate(points)]
    if len(coordinates)>1:
        pygame.draw.lines(surface,GREEN,False,coordinates,1)
    label(surface,f'{low:.4g} .. {high:.4g} / {len(values)} samples',(rect.x+12,rect.bottom-21),12,MUTED,False)
