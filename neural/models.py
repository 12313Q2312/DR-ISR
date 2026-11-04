"""Paper-style polynomial neurons and attributed efficient-kan baseline."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.efficient_kan import KAN


class PolynomialNeuron(nn.Module):
    """sum_k W_k x**k + b, matching the author's polynomial parameterization.

    Search discovers powers; numerical search coefficients become independent
    trainable weights, as in Task_based_neurons/utils/nolinear_neuron.py.
    This implementation avoids exec/eval and retains the author's initialization.
    """
    def __init__(self, n_in, n_out, degrees):
        super().__init__()
        self.degrees = tuple(degrees)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(n_out,n_in)) for _ in degrees])
        self.bias = nn.Parameter(torch.empty(n_out))
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight,a=math.sqrt(5))
        nn.init.uniform_(self.bias,-1/math.sqrt(n_in),1/math.sqrt(n_in))

    def forward(self,x):
        result = self.bias.expand(x.shape[0],-1)
        for degree,weight in zip(self.degrees,self.weights):
            result = result + F.linear(x**degree,weight)
        return result


def build_model(method,n_features,degrees,parameter_cap=256,grid_size=5,spline_order=3):
    """Largest single-hidden-layer width below the SAME parameter ceiling.

    TN: h*(k*(d+1)+1)+1; MLP: h*(d+2)+1.
    KAN uses the vendor's base, spline and independent spline-scaling parameters.
    Widths can differ and actual parameter counts are always reported.
    """
    if method in ("tn_vsr","tn_dr"):
        per_hidden = len(degrees)*(n_features+1)+1
        width = max(1,(parameter_cap-1)//per_hidden) if degrees else 1
        model = nn.Sequential(PolynomialNeuron(n_features,width,degrees),nn.Sigmoid(),
                              PolynomialNeuron(width,1,degrees))
    elif method == "mlp":
        width=max(1,(parameter_cap-1)//(n_features+2))
        model=nn.Sequential(nn.Linear(n_features,width),nn.Sigmoid(),nn.Linear(width,1))
    elif method == "kan":
        per_hidden=(grid_size+spline_order+2)*(n_features+1)
        width=max(1,parameter_cap//per_hidden)
        model=KAN([n_features,width,1],grid_size=grid_size,spline_order=spline_order)
    else:
        raise ValueError(method)
    count=sum(p.numel() for p in model.parameters())
    if count>parameter_cap:
        raise ValueError("Parameter ceiling too small")
    return model,width,count
