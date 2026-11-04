"""Ordered synchronous collectors. Spawn workers import neither Torch nor Pygame."""
import hashlib
import multiprocessing as mp
import time
import traceback
import os
import sys
from ai.environment import Episode
from ai.training_config import TrainingConfig
from ai.encoders import get_encoder,LEGACY


def episode_seed(master,environment,sequence):
    payload=f'snake-frontier-seeds-v1:{master}:{environment}:{sequence}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little') & (2**63-1)


def make_episode(config,encoder):
    return Episode(config,encoder=None if encoder==LEGACY else get_encoder(encoder))


def worker_entry(connection,config,encoder):
    try:
        episode=make_episode(TrainingConfig.from_dict(config),encoder)
        connection.send(('ok',dict(pid=os.getpid(),pygame_loaded='pygame' in sys.modules,torch_loaded='torch' in sys.modules)))
        while True:
            command,value=connection.recv()
            if command=='close':break
            if command=='reset':result=episode.reset(value)
            elif command=='step':result=(episode.step(value),episode.game.score,episode.game.apples_eaten)
            elif command=='usage':
                from ai.profiling import process_usage
                result=process_usage()
            else:raise ValueError('Unknown environment command')
            connection.send(('ok',result))
    except EOFError:pass
    except BaseException:
        try:connection.send(('error',traceback.format_exc(limit=5)))
        except (OSError,EOFError):pass
    finally:connection.close()


class EnvironmentPool:
    def __init__(self,config,encoder,count,strategy='vectorized',timeout=30.):
        self.count,self.strategy,self.timeout=count,strategy,timeout
        self.connections=[];self.processes=[];self.episodes=[];self.closed=False
        self.worker_info=[]
        self.last_usage=[]
        try:
            if strategy=='vectorized':self.episodes=[make_episode(config,encoder) for _ in range(count)]
            elif strategy=='multiprocessing':
                context=mp.get_context('spawn')
                for _ in range(count):
                    parent,child=context.Pipe()
                    process=context.Process(target=worker_entry,args=(child,config.to_dict(),encoder),daemon=True)
                    try:process.start()
                    except BaseException:parent.close();child.close();raise
                    child.close();self.connections.append(parent);self.processes.append(process)
                for index in range(count):self.worker_info.append(self._receive(index))
            else:raise ValueError('Unsupported collector strategy')
        except BaseException:self.close();raise

    def _receive(self,index):
        connection=self.connections[index]
        deadline=time.monotonic()+self.timeout
        while not connection.poll(min(.05,max(0.,deadline-time.monotonic()))):
            if not self.processes[index].is_alive():raise RuntimeError(f'Environment {index} exited unexpectedly')
            if time.monotonic()>=deadline:raise RuntimeError(f'Environment {index} timed out')
        try:status,result=connection.recv()
        except (EOFError,OSError) as error:raise RuntimeError(f'Environment {index} disconnected') from error
        if status!='ok':raise RuntimeError(f'Environment {index} failed: {result}')
        return result

    def reset(self,index,seed):
        if self.closed:raise RuntimeError('Environment pool is closed')
        if self.episodes:return self.episodes[index].reset(seed)
        self.connections[index].send(('reset',seed));return self._receive(index)

    def step(self,actions):
        """Pairs (environment ID, action); results retain exactly that order."""
        if self.closed:raise RuntimeError('Environment pool is closed')
        if self.episodes:
            result=[]
            for index,action in actions:
                episode=self.episodes[index]
                result.append((episode.step(action),episode.game.score,episode.game.apples_eaten))
            return result
        try:
            for index,action in actions:self.connections[index].send(('step',action))
            return [self._receive(index) for index,_ in actions]
        except (OSError,EOFError) as error:raise RuntimeError('Environment communication failed') from error

    def close(self):
        if self.closed:return
        self.closed=True
        for connection in self.connections:
            try:connection.send(('close',None))
            except (OSError,EOFError):pass
        deadline=time.monotonic()+2.
        for process in self.processes:process.join(timeout=max(0.,deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive():process.terminate()
        for process in self.processes:process.join(timeout=2.)
        for connection in self.connections:connection.close()

    def usage(self):
        if self.episodes:return []
        if self.closed:return self.last_usage
        for connection in self.connections:connection.send(('usage',None))
        self.last_usage=[self._receive(index) for index in range(self.count)]
        return self.last_usage
