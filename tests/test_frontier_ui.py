"""Actual Frontier routing, settings, dashboard caching and optional-AI fallback."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import subprocess
import sys
import time
from pathlib import Path
import pytest


def test_frontier_missing_torch_leaves_arcade_available(tmp_path):
    script='''
import sys
from pathlib import Path
sys.modules['torch']=None
from app import Application
app=Application(Path(sys.argv[1]))
try:
    app.action('open_training_hub');app.draw(.016)
    assert app.screen=='training_hub'
    app.action('open_frontier');app.draw(.016)
    assert app.screen=='neural_unavailable'
    app.action('start');app.draw(.016)
    assert app.screen=='game'
finally:app.close()
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'settings.json')],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_frontier_ui_configuration_training_charts_analysis_arena_and_legacy(tmp_path):
    pytest.importorskip('torch')
    from app import Application
    from ai.laboratory import Laboratory
    from ui.neural_view import NeuralView
    app=Application(tmp_path/'settings.json')
    app.lab_view=NeuralView(Laboratory(),tmp_path/'models')
    try:
        app.action('open_training_hub')
        assert [b.text for b in app.controls][:2]==['LEGACY TRAINING','FRONTIER TRAINING']
        app.action('open_frontier')
        view=app.frontier_view
        assert view.headless and view.refresh==100
        view.text['episodes']='0';view.action('start')
        assert view.worker is None
        view.text.update(episodes='26',episode_cap='2',batch_size='2')
        view.refresh=25
        app.draw(.016)
        app.action('frontier:advanced');app.draw(.016)
        app.action('frontier:advanced_page');app.draw(.016)
        app.action('frontier:advanced')
        app.action('frontier:start')
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            app.update(.016)
            if view.snapshot and view.snapshot.status in ('Completed','Failed','Save failed'):break
            time.sleep(.005)
        assert view.snapshot.status=='Completed',view.status
        assert view.snapshot.publications==3
        app.draw(.016)
        cached=view.cache
        app.draw(.016)
        assert view.cache is cached
        app.action('frontier:analysis')
        deadline=time.monotonic()+10
        while view.analysis.jobs and time.monotonic()<deadline:
            app.update(.016);time.sleep(.005)
        assert len(view.analysis.datasets[0].records)==26
        view.analysis.text['games']='2'
        app.action('frontier:analysis:evaluate')
        deadline=time.monotonic()+10
        while view.analysis.jobs and time.monotonic()<deadline:
            app.update(.016);time.sleep(.005)
        from ai.experiments import ExperimentStore
        experiment=ExperimentStore(tmp_path/'experiments').load(Path(view.snapshot.experiment).name)
        assert any(link['kind']=='evaluation' for link in experiment['links'])
        app.draw(.016)
        app.action('frontier:dashboard')
        app.action('frontier:arena')
        assert app.screen=='arena' and app.arena_view.arena.checkpoint_episode==26
        app.draw(.016)
        app.action('open_training')
        assert app.screen=='training' and app.training_view.page=='config'
        app.draw(.016)
    finally:app.close()
