"""Exact raw-record statistics; linear (type-7) quantiles, population deviation."""
from bisect import bisect_left, insort
from dataclasses import dataclass
import math
from statistics import fmean, pstdev


@dataclass(frozen=True)
class EpisodeRecord:
    run_id: str
    episode: int
    score: float
    length: int | None = None
    reward: float | None = None
    reason: str | None = None
    cumulative_steps: int | None = None
    epsilon: float | None = None
    elapsed: float | None = None
    apples: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Dataset:
    identifier: str
    name: str
    records: tuple[EpisodeRecord,...]
    kind: str = 'training'
    architecture: tuple[int,...] = (11,16,8,3)
    status: str = 'Saved checkpoint'
    checkpoint: str | None = None
    limitations: tuple[str,...] = ()
    configuration: dict | None = None
    checkpoint_episode: int | None = None
    model_identity: dict | None = None

    @property
    def parameter_count(self):
        if self.model_identity:return self.model_identity['parameter_count']
        return sum((a+1)*b for a,b in zip(self.architecture,self.architecture[1:]))


def quantile_sorted(values,p):
    if not values:
        return None
    position=(len(values)-1)*p
    low=math.floor(position)
    high=math.ceil(position)
    return values[low]+(values[high]-values[low])*(position-low)


def describe(values):
    values=sorted(values)
    if not values:
        return dict(count=0,minimum=None,q1=None,median=None,mean=None,q3=None,maximum=None,stddev=None,iqr=None)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Statistics require finite values')
    q1,q3=quantile_sorted(values,.25),quantile_sorted(values,.75)
    return dict(count=len(values),minimum=values[0],q1=q1,median=quantile_sorted(values,.5),
                mean=fmean(values),q3=q3,maximum=values[-1],stddev=pstdev(values),iqr=q3-q1)


def select(records,start=None,end=None,alignment='episode'):
    if alignment not in ('episode','steps'):
        raise ValueError('Unknown alignment')
    if alignment=='steps' and any(r.cumulative_steps is None for r in records):
        raise ValueError('Recorded cumulative steps unavailable; use episode alignment')
    def x(r):return r.episode if alignment=='episode' else r.cumulative_steps
    return tuple(r for r in records if (start is None or x(r)>=start) and (end is None or x(r)<=end))


def groups(records,size=100,alignment='episode'):
    if alignment=='episode' and size not in (25,50,100,250,500) or alignment=='steps' and (type(size) is not int or size<1):
        raise ValueError('Grouping must be 25, 50, 100, 250 or 500')
    if alignment not in ('episode','steps'):
        raise ValueError('Unknown alignment')
    result=[]
    buckets={}
    if alignment=='steps':
        if any(r.cumulative_steps is None for r in records):
            raise ValueError('Recorded cumulative steps unavailable; step alignment is disabled for this dataset')
        # Equal recorded-step bins allow paired distributions at equivalent experience.
        for record in records:
            index=max(0,(record.cumulative_steps-1)//size)
            buckets.setdefault(index,[]).append(record)
    else:
        for record in records:
            buckets.setdefault((record.episode-1)//size,[]).append(record)
    for index,batch in sorted(buckets.items()):
        summary=describe([r.score for r in batch])
        result.append(dict(**summary,episode_start=batch[0].episode,episode_end=batch[-1].episode,
            x_start=index*size+1,x_end=(index+1)*size,
            sample_count=len(batch)))
    return tuple(result)


def rolling(records,window=100):
    if window not in (25,50,100,250,500):
        raise ValueError('Rolling window must be 25, 50, 100, 250 or 500')
    ordered=[]
    total=0.
    result=[]
    for index,record in enumerate(records):
        insort(ordered,record.score)
        total+=record.score
        if index>=window:
            previous=records[index-window].score
            ordered.pop(bisect_left(ordered,previous))
            total-=previous
        result.append((record.episode,total/len(ordered),quantile_sorted(ordered,.5)))
    return tuple(result)


def extrema_points(points,limit=600):
    """Retain each bin's original min/max points, including their exact x coordinates."""
    if len(points)<=limit:
        return tuple(points)
    stride=math.ceil(len(points)/max(1,limit//2))
    sampled=[]
    for index in range(0,len(points),stride):
        batch=points[index:index+stride]
        low=min(range(len(batch)),key=lambda i:batch[i][1])
        high=max(range(len(batch)),key=lambda i:batch[i][1])
        sampled.extend(batch[i] for i in sorted({low,high}))
    return tuple(sampled)


def summary(records):
    result=describe([r.score for r in records])
    def complete(field):
        return bool(records) and all(getattr(r,field) is not None for r in records)
    lengths=[r.length for r in records if r.length is not None]
    reasons=[r.reason for r in records if r.reason is not None]
    result.update(total_apples=sum(r.apples for r in records) if complete('apples') else None,
        mean_length=fmean(lengths) if complete('length') else None,
        median_length=quantile_sorted(sorted(lengths),.5) if complete('length') else None,
        collision_percent=100*sum(x in ('wall_collision','self_collision') for x in reasons)/len(records) if complete('reason') else None,
        stagnation_percent=100*reasons.count('stagnation')/len(records) if complete('reason') else None,
        mean_reward=fmean(r.reward for r in records) if complete('reward') else None,
        best_episode=max(records,key=lambda r:r.score).episode if records else None,
        environment_steps=sum(lengths) if complete('length') else None,
        duration=records[-1].elapsed if records and records[0].episode==1 else None,
        reasons_recorded=len(reasons))
    return result
