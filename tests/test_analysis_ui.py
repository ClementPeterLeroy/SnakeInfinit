"""Real Pygame chart rendering, interactions, async loading and cache invalidation."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import time
import pygame
import pytest
from ai.analysis_stats import Dataset,EpisodeRecord
from ai.history import HistoryStore
from ui.analysis_view import AnalysisView
from ui.components import reset_fonts


def settle(view):
    deadline=time.monotonic()+10
    while view.jobs and time.monotonic()<deadline:
        view.update(.016,False)
        time.sleep(.001)
    assert not view.jobs,view.status


@pytest.fixture
def view(tmp_path):
    pygame.init()
    reset_fonts()
    journal=HistoryStore(tmp_path/'history')
    journal.register('a','Model A',(11,16,8,3),{})
    journal.register('b','Model B',(13,32,3),{})
    for run,n in (('a',200),('b',100)):
        for i in range(1,n+1):
            journal.append_episode(EpisodeRecord(run,i,i%17,10,1,'stagnation',i*10,.05,i*.01,i%17))
    journal.close()
    instance=AnalysisView(tmp_path)
    instance.refresh();settle(instance)
    yield instance
    instance.close();reset_fonts();pygame.quit()


def test_context_preserved_modes_grouping_zoom_scroll_and_overlay(view):
    view.action('model_b');view.action('model_b');settle(view)
    assert view.datasets[1].architecture==(13,32,3)
    view.text.update(start='20',end='150')
    view.action('range');settle(view)
    for mode in ('raw','mean_median','distribution'):
        view.action(mode);settle(view)
        assert view.text['start']=='20' and view.text['end']=='150'
        assert view.indices==[0,1] and len(view.bundle['models'])==2
    view.action('group');settle(view)
    assert view.group==250
    view.action('zoom_in');settle(view)
    assert int(view.text['end'])-int(view.text['start'])<130
    before=int(view.text['start'])
    view.action('right');settle(view)
    assert int(view.text['start'])>before
    view.action('alignment');settle(view)
    assert view.alignment=='steps'
    view.action('alignment');settle(view)
    assert view.alignment=='episode'
    view.action('reset');settle(view)
    assert (view.text['start'],view.text['end'])==('1','200')
    surface=pygame.Surface((1280,720))
    for mode in ('distribution','mean_median','raw'):
        view.action(mode);settle(view)
        view.draw(surface,(400,400))
        assert view.chart.tooltip


def test_paired_candle_hover_selects_each_model(view):
    view.action('model_b');view.action('model_b');settle(view)
    surface=pygame.Surface((1280,720))
    view.draw(surface,(-1,-1))
    hits=tuple(view.chart.hits)
    for model_index in (0,1):
        x,_,group=next(h for h in hits if h[1]==model_index)
        view.draw(surface,(round(x),400))
        assert view.chart.tooltip[0].startswith('AB'[model_index]+':')
        assert f"n={group['count']}" in view.chart.tooltip[0]


def test_export_is_async_and_journal_change_invalidates(view):
    view.text.update(start='26',end='27')
    view.action('range');settle(view)
    view.action('export_raw');settle(view)
    assert len(list((view.profile/'exports').glob('*.csv')))==1
    old=view.repository.signature()
    journal=HistoryStore(view.profile/'history')
    journal.append_episode(EpisodeRecord('a',201,999))
    journal.close()
    assert view.repository.signature()!=old
    view.refresh();settle(view)
    assert view.datasets[0].records[-1].episode==201
    assert view.text['start']=='26'


def test_100k_render_uses_cached_data_with_exact_hover(view):
    data=Dataset('large','100,000',tuple(EpisodeRecord('large',i+1,i%19) for i in range(100000)))
    view.datasets=[data,None]
    view.text.update(start='1',end='100000')
    surface=pygame.Surface((1280,720))
    for mode in ('distribution','mean_median','raw'):
        view.action(mode);settle(view)
        calculations=view.model.calculations
        for _ in range(3):view.draw(surface,(400,400))
        assert view.model.calculations==calculations
        assert view.chart.tooltip and view.bundle['models'][0]['summary']['count']==100000
    view.action('alignment')
    assert view.alignment=='episode' and 'unavailable' in view.status


def test_training_tab_syncs_background_controls_and_contains_display_failure(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from ui.training_view import TrainingView
    pygame.init();reset_fonts()
    pygame.display.set_mode((1280,720))
    training=TrainingView(tmp_path,lambda:None)
    try:
        training.action('analysis')
        settle(training.analysis)
        training.analysis.status='background changed'
        training.analysis.layout()
        training.update(.016)
        assert training.buttons[4:]==training.analysis.buttons
        def fail(*args):raise RuntimeError('fixture chart failure')
        monkeypatch.setattr(training.analysis.chart,'draw',fail)
        training.draw(pygame.Surface((1280,720)))
        assert 'fixture chart failure' in training.analysis.status
    finally:
        training.close();reset_fonts();pygame.quit()
