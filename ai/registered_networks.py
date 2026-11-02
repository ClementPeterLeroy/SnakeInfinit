"""Distance MLP and shared compact CNN, plus resolved observation tensor adapters."""
from array import array
import secrets
import struct
import torch
from torch import nn
from ai.encoders import SpatialObservation,ORDERED


class RegisteredNetwork(nn.Module):
    def __init__(self,spec):
        super().__init__()
        self.config=spec
        self.edited_parameters=set()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec.seed if spec.seed is not None else secrets.randbits(63))
            if spec.configuration_id=='B':
                self.layers=nn.ModuleList(nn.Linear(a,b) for a,b in zip(spec.sizes,spec.sizes[1:]))
            else:
                self.convolutions=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.ReLU(),
                    nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,stride=2,padding=1),nn.ReLU())
                self.hidden=nn.Linear(64*4*4+4,128)
                self.output=nn.Linear(128,3)

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def forward(self,inputs):
        if self.config.configuration_id=='B':
            if inputs.ndim not in (1,2) or inputs.shape[-1]!=19:raise ValueError('Distance MLP expects 19 features')
            if inputs.device.type!='cpu' or not torch.isfinite(inputs).all():raise ValueError('Finite CPU inputs required')
            for index,layer in enumerate(self.layers):
                inputs=layer(inputs)
                if index<len(self.layers)-1:inputs=torch.relu(inputs)
            return inputs
        if not isinstance(inputs,dict) or set(inputs)!= {'spatial','heading'}:raise ValueError('CNN requires spatial and heading tensors')
        board,heading=inputs['spatial'],inputs['heading']
        single=board.ndim==3
        if single:board,heading=board.unsqueeze(0),heading.unsqueeze(0)
        if board.ndim!=4 or tuple(board.shape[1:])!=(3,25,25) or heading.shape!=(board.shape[0],4):raise ValueError('CNN input shape mismatch')
        if board.device.type!='cpu' or heading.device.type!='cpu' or not torch.isfinite(board).all() or not torch.isfinite(heading).all():raise ValueError('Finite CPU inputs required')
        features=self.convolutions(board).flatten(1)
        values=self.output(torch.relu(self.hidden(torch.cat((features,heading),dim=1))))
        return values[0] if single else values


class ObservationCodec:
    """Resolved once per model/session; compact replay never flattens heading into a grid."""
    def __init__(self,spec):
        self.spec=spec
        self.spatial=spec.configuration_id in ('C','D')
        self.ordered=spec.encoder_id==ORDERED

    def stack(self,observations):
        if not self.spatial:return torch.tensor(observations,dtype=torch.float32)
        values=array('f',[0.])*(len(observations)*1875)
        for index,observation in enumerate(observations):observation.write(values,index*1875)
        return dict(spatial=torch.frombuffer(values,dtype=torch.float32).reshape(-1,3,25,25),
            heading=torch.tensor([obs.heading for obs in observations],dtype=torch.float32).reshape(-1,4))

    def single(self,observation):
        if not self.spatial:return torch.tensor(observation,dtype=torch.float32)
        batch=self.stack((observation,))
        return {name:value[0] for name,value in batch.items()}

    def pack(self,observations):
        if not self.spatial:return torch.tensor(observations,dtype=torch.float32).reshape(-1,19)
        return dict(positions=[obs.positions for obs in observations],foods=[obs.food for obs in observations],
            headings=[obs.heading for obs in observations],ordered=self.ordered)

    def unpack(self,data,n):
        if not self.spatial:
            if not isinstance(data,torch.Tensor) or data.dtype!=torch.float32 or data.shape!=(n,19) or not torch.isfinite(data).all() or (data<0).any() or (data>1).any() or ((data[:,:11]!=0)&(data[:,:11]!=1)).any():
                raise ValueError('Invalid distance replay observations')
            return [tuple(row) for row in data.tolist()]
        if not isinstance(data,dict) or set(data)!= {'positions','foods','headings','ordered'} or data['ordered'] is not self.ordered:
            raise ValueError('Invalid spatial replay encoding')
        if any(not isinstance(data[key],list) or len(data[key])!=n for key in ('positions','foods','headings')):raise ValueError('Invalid spatial replay length')
        result=[]
        for positions,food,heading in zip(data['positions'],data['foods'],data['headings']):
            if not isinstance(positions,bytes) or not 2<=len(positions)<=1250 or len(positions)%2:raise ValueError('Invalid spatial positions')
            cells=struct.unpack('<'+'H'*(len(positions)//2),positions)
            if any(cell>=625 for cell in cells) or len(set(cells))!=len(cells):raise ValueError('Invalid spatial occupancy')
            if type(food) is not int or not -1<=food<625:raise ValueError('Invalid spatial food')
            if not isinstance(heading,(list,tuple)) or len(heading)!=4 or any(x not in (0.,1.) for x in heading) or sum(heading)!=1:raise ValueError('Invalid heading')
            result.append(SpatialObservation(positions,food,tuple(heading),self.ordered))
        return result
