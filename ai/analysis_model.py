"""Cached derived chart data prepared off the render thread from exact records."""
from collections import OrderedDict
from ai.analysis_stats import select,summary,groups,rolling,extrema_points


class AnalysisModel:
    def __init__(self):
        self.cache=OrderedDict()
        self.calculations=0

    def prepare(self,datasets,mode='distribution',size=100,alignment='episode',start=None,end=None,step_bin=25000):
        if mode not in ('distribution','mean_median','raw'):
            raise ValueError('Unknown score visualization mode')
        if len({d.kind for d in datasets})>1:
            raise ValueError('Compare training with training, or evaluation with evaluation; do not mix datasets')
        if len(datasets)>1 and datasets[0].kind=='evaluation':
            first=datasets[0].configuration or {}
            if any(not first or d.configuration!=first for d in datasets[1:]):
                raise ValueError('Evaluation comparison requires identical recorded seed lists and settings')
        key=(tuple((d.identifier,id(d.records)) for d in datasets),mode,size,alignment,start,end,step_bin)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self.calculations+=1
        results=[]
        for dataset in datasets:
            chosen=select(dataset.records,start,end,alignment)
            stats=summary(chosen)
            xs=lambda record:record.episode if alignment=='episode' else record.cumulative_steps
            if mode=='distribution':
                series=groups(chosen,size if alignment=='episode' else step_bin,alignment)
            elif mode=='mean_median':
                # Compute the actual trailing window before clipping the visible range.
                values=rolling(dataset.records,size)
                by_episode={record.episode:record for record in chosen}
                series=tuple((xs(by_episode[episode]),mean,median,episode) for episode,mean,median in values if episode in by_episode)
            else:
                series=tuple((xs(record),record.score,record.episode) for record in chosen)
            if mode=='distribution':
                import math
                stride=max(1,math.ceil(len(series)/140))
                rendered=[]
                for offset in range(0,len(series),stride):
                    batch=series[offset:offset+stride]
                    for item in (min(batch,key=lambda g:g['minimum']),max(batch,key=lambda g:g['maximum'])):
                        if not rendered or rendered[-1] is not item:rendered.append(item)
                render=tuple(rendered)
            else:
                render={column:extrema_points(tuple((row[0],row[column]) for row in series),600)
                        for column in ((1,2) if mode=='mean_median' else (1,))}
            bounds=(min(xs(r) for r in chosen),max(xs(r) for r in chosen),min(r.score for r in chosen),max(r.score for r in chosen)) if chosen else None
            results.append(dict(dataset=dataset,records=chosen,summary=stats,series=series,render=render,bounds=bounds,
                recorded_steps=max((r.cumulative_steps for r in dataset.records if r.cumulative_steps is not None),default=None)))
        bundle=dict(mode=mode,size=size,alignment=alignment,start=start,end=end,step_bin=step_bin,models=tuple(results))
        self.cache[key]=bundle
        while len(self.cache)>4:self.cache.popitem(last=False)
        return bundle
