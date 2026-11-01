"""These checks run even in arcade-only environments without Torch installed."""
import os
import subprocess
import sys


def test_arcade_startup_with_torch_entirely_blocked(tmp_path):
    script = '''
import builtins, sys
old = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] == 'torch': raise AssertionError('Unexpected torch import')
    return old(name, *args, **kwargs)
builtins.__import__ = guard
sys.modules['torch'] = None
from app import Application
from pathlib import Path
from core.challenges import BY_ID
app = Application(Path(sys.argv[1]))
try:
    app.action('ai'); app.draw(.016)
    app.action('open_neural')
    assert app.screen == 'neural_unavailable'
    app.draw(.016)
    app.action('training'); app.draw(.016)
    for mode in ('classic', 'arcade', *BY_ID):
        app.action('play:' + mode)
        for i in range(3): app.update(1)
        app.update(.15); app.draw(.016)
    assert not any(name.startswith('torch.') for name in sys.modules)
finally:
    app.close()
'''
    environment = dict(os.environ, SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'settings.json')],
                            capture_output=True, text=True, timeout=30, env=environment)
    assert result.returncode == 0, result.stderr


def test_all_three_ai_entries_gracefully_handle_missing_torch(tmp_path):
    script='''
import sys
sys.modules['torch']=None
from pathlib import Path
from app import Application
app=Application(Path(sys.argv[1]))
try:
    for action in ('open_neural','open_training','open_arena'):
        app.action(action)
        assert app.screen=='neural_unavailable'
        app.draw(.016)
finally:app.close()
'''
    env=dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'settings.json')],
                          capture_output=True,text=True,timeout=30,env=env)
    assert result.returncode==0,result.stderr
