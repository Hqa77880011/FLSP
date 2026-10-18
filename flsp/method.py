import numpy as np
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.get('input_shape',[8])[-1];self.net=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,b):return {'prediction':self.net(b['x'].float())}
    def objective(self,b,p):return F.mse_loss(p['prediction'],b['y'].float())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,8)).astype('float32');return {'x':x,'y':(x.sum(1,keepdims=True)/8).astype('float32'),'group':np.arange(n)}


def placement(problem,variant='full'):
    import numpy as np
    groups=problem['groups'];weights=np.asarray(problem['weights'],float);weights=weights/weights.sum();selected=[0]*len(groups);caps=np.asarray(problem['caps']);alpha=problem.get('unavailability_penalty',1.)
    def summarize(selection):
        risk=0.;unavailable=0.;cost=np.asarray(problem['base_cost'],float).copy();scratch=0.;recover=0.;traffic=0.
        for g,index in enumerate(selection):
            action=groups[g][index];risk+=weights@np.asarray(action['risk']);unavailable+=weights@np.asarray(action['unavailable']);cost+=action['cost'];scratch=max(scratch,action.get('scratch',0));invoke=weights@np.asarray(action.get('invoke',np.zeros(len(weights))));recover+=invoke*action.get('recovery_time',0);traffic+=invoke*action.get('recovery_traffic',0)
        cost[2]+=scratch;cost[0]+=problem.get('intensity',0)*recover;cost[1]+=problem.get('intensity',0)*traffic
        return risk,unavailable,cost
    while True:
        risk,unavailable,cost=summarize(selected);best=None
        for g,actions in enumerate(groups):
            for index,action in enumerate(actions):
                if index==selected[g] or not action.get('compatible',True):continue
                proposal=selected.copy();proposal[g]=index;r,u,c=summarize(proposal)
                if np.any(c>caps) or u>problem.get('max_unavailable',.025):continue
                reduction=(risk+alpha*unavailable)-(r+alpha*u)
                if variant in {'baseline','cost_aware'}:reduction=action.get('protected_exposure',0)-actions[selected[g]].get('protected_exposure',0)
                denominator=np.maximum(c-cost,0).dot(1/caps);score=reduction/max(denominator,1e-12)
                if reduction>0 and (best is None or score>best[0]):best=(score,proposal)
        if best is None:break
        selected=best[1]
    risk,unavailable,cost=summarize(selected);base_risk=summarize([0]*len(groups))[0]
    return {'actions':selected,'conditional_sdc':risk,'unavailability':unavailable,'risk_weighted_coverage':1-risk/base_risk if base_risk>0 else None,'all_inference_sdc':float(-np.expm1(-problem.get('intensity',0)*risk)),'cost':cost.tolist(),'feasible':bool(np.all(cost<=caps) and unavailable<=problem.get('max_unavailable',.025))}


def execute(args):
    import json
    from pathlib import Path
    problem=json.loads(Path(args.input).read_text());result=placement(problem,args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'placement.json').write_text(json.dumps(result,indent=2))
