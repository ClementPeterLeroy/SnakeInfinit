"""Deterministic observation registry, independent of Torch and presentation."""
from array import array
from dataclasses import dataclass
import struct
from ai.observations import encode_observation,FEATURE_NAMES
from core.types import Direction

LEGACY='snake-classic-binary11-v1'
DISTANCE='snake-distance19-v1'
BINARY='snake-spatial-binary-v1'
ORDERED='snake-spatial-ordered-v1'
HEADING=(Direction.UP,Direction.DOWN,Direction.LEFT,Direction.RIGHT)
COMPASS=((0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1))
DISTANCE_NAMES=('North','Northeast','East','Southeast','South','Southwest','West','Northwest')


@dataclass(frozen=True)
class SpatialObservation:
    """Lossless compact spatial record; dense float32 channels materialize on demand."""
    positions: bytes  # little-endian uint16 y*25+x, head then ordered body
    food: int
    heading: tuple
    ordered: bool

    def write(self,buffer,offset=0):
        positions=struct.unpack('<'+'H'*(len(self.positions)//2),self.positions)
        n=len(positions)-1
        for index,cell in enumerate(positions[1:]):
            buffer[offset+cell]=(n-index)/n if self.ordered else 1.
        buffer[offset+625+positions[0]]=1.
        if self.food>=0:buffer[offset+1250+self.food]=1.

    @property
    def spatial(self):
        values=array('f',[0.])*1875
        self.write(values)
        return tuple(tuple(tuple(values[c*625+y*25:c*625+(y+1)*25]) for y in range(25)) for c in range(3))


def distance19(state):
    _board(state)
    x,y=state.snake[0]
    occupied=set(state.snake[1:])  # Static occupancy including the tail; food is free.
    result=[]
    for dx,dy in COMPASS:
        count=0
        px,py=x+dx,y+dy
        while 0<=px<25 and 0<=py<25 and (px,py) not in occupied:
            count+=1;px+=dx;py+=dy
        result.append(count/24.)
    return tuple(array('f',(*encode_observation(state),*result)))


def _board(state):
    if (state.board_width,state.board_height)!=(25,25) or not state.snake:
        raise ValueError('This encoder requires a nonempty Snake on a 25x25 board')


def spatial(state,ordered=False):
    _board(state)
    cells=tuple(y*25+x for x,y in state.snake)
    direction=state.pending_direction or state.direction
    return SpatialObservation(struct.pack('<'+'H'*len(cells),*cells),
        -1 if state.apple is None else state.apple[1]*25+state.apple[0],
        tuple(float(direction==heading) for heading in HEADING),ordered)


@dataclass(frozen=True)
class Encoder:
    identifier: str
    name: str
    shape: object
    ordering: tuple
    normalization: str
    architectures: tuple
    encode: object
    dtype: str = 'float32'
    schema: int = 1


ENCODERS={
    LEGACY:Encoder(LEGACY,'Original 11 features',(11,),FEATURE_NAMES,'Original binary indicators',('mlp_11_16_8_3',),encode_observation),
    DISTANCE:Encoder(DISTANCE,'Eight-direction distance sensing',(19,),(*FEATURE_NAMES,*DISTANCE_NAMES),
        'Consecutive free cells / 24; static body including tail; food is not an obstacle',('mlp_distance19',),distance19),
    BINARY:Encoder(BINARY,'Binary body occupancy',{'spatial':(3,25,25),'heading':(4,)},('body','head','food','heading U,D,L,R'),
        'Body excluding head = 1; empty = 0',('cnn_spatial_compact',),lambda state:spatial(state,False)),
    ORDERED:Encoder(ORDERED,'Ordered body positions',{'spatial':(3,25,25),'heading':(4,)},('ordered body','head','food','heading U,D,L,R'),
        'Body index i=0..N-1: (N-i)/N; no head in body; empty = 0',('cnn_spatial_compact',),lambda state:spatial(state,True)),
}


def get_encoder(identifier):
    try:return ENCODERS[identifier]
    except (KeyError,TypeError):raise ValueError('Unsupported observation encoder: '+str(identifier)) from None
