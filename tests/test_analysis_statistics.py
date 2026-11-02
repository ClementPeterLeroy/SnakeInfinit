"""Exact raw statistics, heterogeneous models, alignment and 100k-record caching."""
import math
import time
import pytest
from ai.analysis_stats import Dataset,EpisodeRecord,describe,groups,rolling,select,summary,extrema_points
from ai.analysis_model import AnalysisModel


def records(values,run='a',steps=True):
    return tuple(EpisodeRecord(run,i+1,value,10,value*2,'wall_collision' if i%2 else 'stagnation',
        (i+1)*10 if steps else None,.1,(i+1)*.5,int(value)) for i,value in enumerate(values))


def test_linear_quartiles_population_deviation_and_iqr():
    result=describe([1,2,3,4])
    assert result==dict(count=4,minimum=1,q1=1.75,median=2.5,mean=2.5,q3=3.25,maximum=4,
                       stddev=math.sqrt(1.25),iqr=1.5)


@pytest.mark.parametrize('values',([], [7], [2,4], [0,0,0,0]))
def test_empty_and_small_samples(values):
    result=describe(values)
    if not values:assert result['count']==0 and result['mean'] is None
    else:
        assert result['minimum']==min(values) and result['maximum']==max(values)
        assert result['mean']==sum(values)/len(values)
        if len(set(values))==1:assert result['stddev']==0 and result['iqr']==0


def test_incomplete_groups_have_true_counts_and_absolute_episode_ranges():
    data=records(list(range(27)))
    candles=groups(data,25)
    assert len(candles)==2 and candles[1]['sample_count']==2
    assert candles[1]['episode_start']==26 and candles[1]['episode_end']==27
    assert candles[1]['minimum']==25 and candles[1]['maximum']==26
    assert candles[1]['q1']==25.25 and candles[1]['q3']==25.75
    assert groups((),100)==()


def test_rolling_uses_raw_scores_and_exact_medians():
    values=[0]*24+[100,200]
    result=rolling(records(values),25)
    assert result[24]==(25,4.,0.)
    assert result[25]==(26,12.,0.)
    assert result[0]==(1,0.,0.)


def test_summary_uses_full_raw_data_and_marks_missing_fields():
    complete=summary(records([0,1,2,3]))
    assert complete['total_apples']==6 and complete['collision_percent']==50
    assert complete['stagnation_percent']==50 and complete['mean_length']==10
    assert complete['duration']==2. and complete['mean_reward']==3.
    missing=summary((EpisodeRecord('old',500,9),))
    assert missing['mean']==9 and missing['total_apples'] is None
    assert missing['collision_percent'] is None and missing['environment_steps'] is None
    legacy=(EpisodeRecord('old',1,0),EpisodeRecord('old',2,4,elapsed=15.5))
    assert summary(legacy)['duration']==15.5


def test_comparison_different_architectures_lengths_and_actual_step_alignment():
    a=Dataset('a','Small',records([1,3,4]),architecture=(11,16,8,3))
    b=Dataset('b','Future metadata',records([4,0],'b'),architecture=(13,32,3))
    model=AnalysisModel()
    bundle=model.prepare((a,b),'raw',alignment='steps',start=15,end=30)
    assert bundle['models'][0]['series']==((20,3,2),(30,4,3))
    assert bundle['models'][1]['series']==((20,0,2),)
    assert b.parameter_count==(13+1)*32+(32+1)*3
    assert a.parameter_count==355
    assert model.prepare((a,b),'raw',alignment='steps',start=15,end=30) is bundle
    assert model.calculations==1


def test_missing_steps_and_mixed_evaluation_are_rejected():
    a=Dataset('a','Legacy',records([1,2],steps=False))
    with pytest.raises(ValueError,match='steps unavailable'):
        AnalysisModel().prepare((a,),'raw',alignment='steps')
    with pytest.raises(ValueError,match='do not mix'):
        AnalysisModel().prepare((a,Dataset('e','Evaluation',records([3]),kind='evaluation')))


def test_extrema_preserved_and_100000_record_derivations_cached():
    points=tuple((i,1000 if i==54321 else -500 if i==12345 else 0) for i in range(100000))
    reduced=extrema_points(points)
    assert (54321,1000) in reduced and (12345,-500) in reduced and len(reduced)<=600
    dataset=Dataset('big','100k',records([i%23 for i in range(100000)]))
    model=AnalysisModel()
    start=time.perf_counter()
    result=model.prepare((dataset,),'mean_median')
    seconds=time.perf_counter()-start
    assert result['models'][0]['summary']['count']==100000
    assert len(result['models'][0]['series'])==100000
    assert len(result['models'][0]['render'][1])<=600
    assert model.prepare((dataset,),'mean_median') is result and model.calculations==1
    assert seconds<15  # Loose guard against accidental per-point quadratic work.
