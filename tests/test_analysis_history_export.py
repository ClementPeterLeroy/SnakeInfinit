"""Versioned external history, precise UTF-8 CSV and future-ready experiment metadata."""
import csv
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from ai.analysis_stats import Dataset,EpisodeRecord
from ai.history import HistoryStore
from ai.analysis_export import export_csv
from ai.experiments import Experiment,ExperimentStore


def test_journal_append_read_only_discovery_duplicate_conflict_and_no_migration(tmp_path):
    store=HistoryStore(tmp_path/'history')
    assert store.datasets()==() and not store.path.exists()
    store.register('run','模型 é',[11,16,8,3],{'seed':0})
    row=EpisodeRecord('run',1,2,30,19.7,'wall_collision',30,.9,1.0,2)
    store.append_optimization('run',30,.4,.2,.9)
    store.append_episode(row)
    store.append_episode(row)
    data=store.load('run')
    assert data.records==(row,) and data.name=='模型 é'
    with pytest.raises(ValueError,match='conflict'):
        store.append_episode(EpisodeRecord('run',1,99))
    store.status('run','Completed')
    assert store.load('run').status=='Completed'
    store.close()


def test_csv_raw_comparison_unicode_missing_values_range_and_no_overwrite(tmp_path):
    a=Dataset('a','模型 Ω',(EpisodeRecord('a',1,1),EpisodeRecord('a',2,5)))
    b=Dataset('b','Émeraude',(EpisodeRecord('b',2,3,10),))
    path=export_csv(tmp_path/'raw.csv',(a,b),start=2,end=2)
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['model_name']=='模型 Ω'
    assert rows[0]['episode']=='2' and rows[0]['length']=='' and rows[1]['length']=='10'
    assert json.loads(path.with_suffix('.csv.metadata.json').read_text())['selected_range']==[2,2]
    with pytest.raises(FileExistsError):export_csv(path,(a,))


def test_grouped_csv_uses_actual_grouping_and_population_statistics(tmp_path):
    dataset=Dataset('a','A',tuple(EpisodeRecord('a',i+1,i) for i in range(27)))
    path=export_csv(tmp_path/'groups.csv',(dataset,),'grouped',size=25)
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert rows[1]['sample_count']=='2' and rows[1]['q1']=='25.25'
    assert rows[1]['minimum']=='25' and rows[1]['maximum']=='26'


def test_csv_file_errors_are_reported(tmp_path):
    with patch.object(Path,'open',side_effect=PermissionError('read-only')):
        with pytest.raises(PermissionError):export_csv(tmp_path/'blocked.csv',())


def test_experiment_architecture_is_generic_and_metadata_versioned(tmp_path):
    experiment=Experiment('test123','Future descriptor','external-encoder',19,(19,32,3),{'gamma':.99},
                          {'apple':10},7,(1,2,3),100000,('run',),('checkpoint',))
    store=ExperimentStore(tmp_path)
    store.save(experiment)
    store.link('test123','evaluation','eval.json')
    data=store.load('test123')
    assert data['input_count']==19 and data['schema_version']==1
    assert data['links'][0]['kind']=='evaluation'
    with pytest.raises(FileExistsError):store.save(experiment)
    with pytest.raises(ValueError):
        Experiment('bad','Bad','encoder',13,(11,8,3),{}, {},0,(),100)
