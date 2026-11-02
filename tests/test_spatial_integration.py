"""All four configurations through the actual Frontier/Arena/Analysis workflows."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
from pathlib import Path
import time
import json
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(1)
from ai.model_registry import model_spec,identity
from ai.frontier import FrontierWorker
from ai.frontier_store import load_published_model
from ai.training_config import TrainingConfig
from ai.analysis_repository import AnalysisRepository
from ai.analysis_model import AnalysisModel
from ai.analysis_export import export_named_csv
from ai.evaluation import evaluate_checkpoints


def complete(worker):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        snapshot=worker.latest()
        if snapshot and snapshot.status in ('Completed','Failed','Save failed'):return snapshot
        time.sleep(.005)
    pytest.fail('Frontier timed out')


@pytest.mark.parametrize('configuration','ABCD')
def test_frontier_all_configurations_evaluation_and_exports(configuration,tmp_path):
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'research')
    try:
        spec=model_spec(configuration,7)
        worker.start(TrainingConfig(episodes=3,episode_cap=3,batch_size=2),architecture=spec)
        result=complete(worker)
        assert result.status=='Completed',result.message
        model,metadata=load_published_model(result.model)
        assert metadata['model_configuration_id']==configuration
        assert metadata['episode']==3 and metadata['environment_steps']==9
        assert metadata['parameter_count']==identity(spec)['parameter_count']
        repo=AnalysisRepository(tmp_path/'profile')
        choice=next(c for c in repo.choices() if str(c.source)==result.checkpoint)
        dataset=repo.load(choice)
        assert dataset.parameter_count==metadata['parameter_count'] and len(dataset.records)==3
        assert dataset.model_identity['encoder_id']==spec.encoder_id
        assert export_named_csv(tmp_path/'exports',(dataset,)).is_file()
        evaluated=evaluate_checkpoints([Path(result.checkpoint)],(0,1),tmp_path/'profile/evaluations')
        assert json.loads(evaluated.read_text())['models'][0]['model_identity']['encoder_id']==spec.encoder_id
        repeated=evaluate_checkpoints([Path(result.checkpoint)],(0,1),tmp_path/'profile/evaluations')
        assert json.loads(evaluated.read_text())['models']==json.loads(repeated.read_text())['models']
    finally:worker.shutdown()


def test_frontier_ui_lineup_and_cross_family_analysis(tmp_path):
    from app import Application
    from ai.laboratory import Laboratory
    from ui.neural_view import NeuralView
    app=Application(tmp_path/'settings.json')
    app.lab_view=NeuralView(Laboratory(),tmp_path/'models')
    checkpoints=[]
    try:
        app.action('open_frontier')
        view=app.frontier_view
        view.text.update(episodes='2',episode_cap='2',batch_size='2')
        for configuration in 'ABCD':
            assert view.model_configuration==configuration
            app.action('frontier:config');app.draw(.016)
            app.action('frontier:start')
            result=complete(view.worker)
            assert result.status=='Completed',result.message
            app.update(.016);app.draw(.016)
            checkpoints.append(Path(result.checkpoint))
            app.action('frontier:arena');app.draw(.016)
            assert app.arena_view.arena.network.config.input_size==(11 if configuration=='A' else 19 if configuration=='B' else 1879)
            app.action('open_frontier');app.action('frontier:config')
            app.action('frontier:model_configuration')
        result=evaluate_checkpoints(checkpoints[2:],(7,8),tmp_path/'evaluations')
        repository=AnalysisRepository(tmp_path)
        datasets=tuple(repository.load(c) for c in repository.choices() if c.source==result)
        assert len(datasets)==2 and datasets[0].parameter_count==datasets[1].parameter_count==188419
        assert AnalysisModel().prepare(datasets)['models'][0]['summary']['count']==2
        assert export_named_csv(tmp_path/'exports',datasets,'grouped').is_file()
    finally:app.close()
