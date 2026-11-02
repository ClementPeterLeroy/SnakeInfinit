"""Large score charts with exact hover data and explicit render-only decimation."""
import math
import pygame
from ai.analysis_stats import extrema_points
from ui.components import PANEL,BORDER,TEXT,GREEN,MUTED,label

MODEL_COLORS=(GREEN,(155,174,227))


class AnalysisChart:
    def __init__(self):
        self.hits=[]
        self.tooltip=()

    def draw(self,surface,rect,bundle,mouse,show_mean=True,show_median=True):
        self.hits=[]
        self.tooltip=()
        pygame.draw.rect(surface,PANEL,rect,border_radius=7)
        if not bundle or not any(m['records'] for m in bundle['models']):
            label(surface,'Select a dataset and range with available records',rect.center,18,MUTED)
            return
        models=bundle['models']
        alignment=bundle['alignment']
        mode=bundle['mode']
        bounds=[model['bounds'] for model in models if model['bounds'] is not None]
        xmin,xmax=min(b[0] for b in bounds),max(b[1] for b in bounds)
        ymin,ymax=min(0,min(b[2] for b in bounds)),max(b[3] for b in bounds)
        ymax=max(ymin+1,ymax*1.05)
        plot=pygame.Rect(rect.x+55,rect.y+31,rect.width-76,rect.height-72)
        X=lambda value:plot.x+(value-xmin)*plot.width/max(1,xmax-xmin)
        Y=lambda value:plot.bottom-(value-ymin)*plot.height/(ymax-ymin)
        for index in range(5):
            value=ymin+(ymax-ymin)*index/4
            y=round(Y(value))
            pygame.draw.line(surface,BORDER,(plot.left,y),(plot.right,y))
            label(surface,f'{value:.3g}',(plot.x-8,y),12,MUTED,True)
        for index in range(5):
            value=xmin+(xmax-xmin)*index/4
            label(surface,f'{value:,.0f}',(X(value),plot.bottom+15),12,MUTED)
        subtitle='true min/max whiskers; linear Q1/median/Q3' if mode=='distribution' else 'raw trailing mean / median' if mode=='mean_median' else 'individual episode scores / extrema-preserving display'
        label(surface,subtitle,(rect.x+12,rect.y+7),12,MUTED,False)
        old=surface.get_clip()
        surface.set_clip(plot.inflate(2,2))
        for model_index,model in enumerate(models):
            color=MODEL_COLORS[model_index]
            series=model['series']
            if mode=='distribution':
                # Exact candle summaries remain available; dense full-history views
                # display a subset of original candles rather than averaging quartiles.
                for group in model['render']:
                    center=(max(xmin,group['x_start'])+min(xmax,group['x_end']))/2
                    cell=max(1,plot.width*(group['x_end']-group['x_start']+1)/max(1,xmax-xmin))
                    width=max(1,min(12,cell*.32))
                    x=X(center)+(model_index-.5)*width*1.4 if len(models)==2 else X(center)
                    pygame.draw.line(surface,color,(x,Y(group['minimum'])),(x,Y(group['maximum'])),1)
                    for value in (group['minimum'],group['maximum']):
                        pygame.draw.line(surface,color,(x-width*.45,Y(value)),(x+width*.45,Y(value)),1)
                    box=pygame.Rect(x-width/2,Y(group['q3']),width,max(1,Y(group['q1'])-Y(group['q3'])))
                    pygame.draw.rect(surface,color,box,1)
                    pygame.draw.line(surface,color,(x-width/2,Y(group['median'])),(x+width/2,Y(group['median'])),2)
                    if show_mean:pygame.draw.circle(surface,color,(round(x),round(Y(group['mean']))),2)
                    self.hits.append((x,model_index,group))
            else:
                columns=(1,2) if mode=='mean_median' else (1,)
                for column in columns:
                    if mode=='mean_median' and (column==1 and not show_mean or column==2 and not show_median):continue
                    points=model['render'][column]
                    coordinates=[(X(x),Y(y)) for x,y in points]
                    if len(coordinates)>1:
                        if mode=='mean_median' and column==2:
                            for index in range(1,len(coordinates),2):
                                pygame.draw.line(surface,color,coordinates[index-1],coordinates[index],1)
                        else:pygame.draw.lines(surface,color,False,coordinates,1)
                    elif coordinates:pygame.draw.circle(surface,color,tuple(map(round,coordinates[0])),2)
        surface.set_clip(old)
        label(surface,'Recorded environment steps' if alignment=='steps' else 'Episode',(plot.centerx,rect.bottom-12),12,MUTED)
        if rect.collidepoint(mouse):
            if mode=='distribution' and self.hits:
                from bisect import bisect_left
                target=xmin+(mouse[0]-plot.x)*max(1,xmax-xmin)/plot.width
                candidates=[]
                for model_index,model in enumerate(models):
                    series=model['series']
                    if not series:continue
                    index=bisect_left(series,target,key=lambda g:(g['x_start']+g['x_end'])/2)
                    for group in series[max(0,index-1):min(len(series),index+1)]:
                        center=(max(xmin,group['x_start'])+min(xmax,group['x_end']))/2
                        cell=max(1,plot.width*(group['x_end']-group['x_start']+1)/max(1,xmax-xmin))
                        width=max(1,min(12,cell*.32))
                        x=X(center)+(model_index-.5)*width*1.4 if len(models)==2 else X(center)
                        candidates.append((abs(x-mouse[0]),model_index,group))
                _,model_index,group=min(candidates,key=lambda item:item[0])
                self.tooltip=(f"{'AB'[model_index]}: episodes {group['episode_start']}–{group['episode_end']} / n={group['count']}",
                    f"min {group['minimum']:g}  Q1 {group['q1']:.4g}  median {group['median']:.4g}  mean {group['mean']:.4g}",
                    f"Q3 {group['q3']:.4g}  max {group['maximum']:g}  population SD {group['stddev']:.4g}")
            elif mode!='distribution':
                lines=[]
                target=xmin+(mouse[0]-plot.x)*max(1,xmax-xmin)/plot.width
                from bisect import bisect_left
                for model_index,model in enumerate(models):
                    series=model['series']
                    if not series:continue
                    # Binary search exact, full-resolution data, not rendered bins.
                    index=bisect_left(series,target,key=lambda row:row[0])
                    row=min(series[max(0,index-1):min(len(series),index+1)],key=lambda r:abs(r[0]-target))
                    lines.append(f"{'AB'[model_index]}: "+(f"episode {row[3]}, mean {row[1]:.5g}, median {row[2]:.5g}" if mode=='mean_median' else f"episode {row[2]}, score {row[1]:g}, x={row[0]}"))
                self.tooltip=tuple(lines)
