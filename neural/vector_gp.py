"""Homogeneous polynomial VSR: one scalar syntax applied to all coordinates.

    f(X) = mean_j p(X_j). Mean vs sum only changes coefficient scale.
    Both search arms use exactly this space, initialization and charged budget.
    This is a controlled reimplementation, not the author's unmodified GP script.
"""

from dataclasses import asdict,dataclass
import hashlib
import json
from time import perf_counter
import numpy as np
from dr_isr.tree import Node


@dataclass(frozen=True)
class SearchConfig:
    budget:int=1500
    population:int=48
    init_depth:int=3
    max_depth:int=5
    max_nodes:int=47
    max_degree:int=8
    guide_candidates:int=4
    tournament:int=5
    crossover_rate:float=.3
    mutation_rate:float=.6
    parsimony:float=1e-5

    def __post_init__(self):
        if self.budget<self.population or self.population<2 or self.max_degree<1:
            raise ValueError("Invalid search budget/population/degree")


def polynomial(tree):
    if tree.op=="x":return np.array([0.,1.])
    if tree.op=="const":return np.array([float(tree.value)])
    a,b=map(polynomial,tree.children)
    if tree.op=="add":out=np.polynomial.polynomial.polyadd(a,b)
    elif tree.op=="sub":out=np.polynomial.polynomial.polysub(a,b)
    elif tree.op=="mul":out=np.polynomial.polynomial.polymul(a,b)
    else:raise ValueError("Nonpolynomial operator")
    return np.trim_zeros(out,"b") if np.any(out) else np.array([0.])


def valid(tree,cfg):
    if tree.depth>cfg.max_depth or tree.size>cfg.max_nodes:return False
    coefficients=polynomial(tree)
    return len(coefficients)>1 and len(coefficients)<=cfg.max_degree+1 and np.max(np.abs(coefficients))<=1000


def random_tree(rng,depth):
    if depth==0 or rng.random()<.3:
        return Node("x",value=0) if rng.random()<.7 else Node("const",value=float(rng.integers(-2,3)))
    op=str(rng.choice(["add","sub","mul"]))
    return Node(op,(random_tree(rng,depth-1),random_tree(rng,depth-1)))


def evaluate(tree,x,cache=None,path=()):
    if tree.op=="x":out=x
    elif tree.op=="const":out=np.full(x.shape,tree.value,dtype=float)
    else:
        a,b=[evaluate(c,x,cache,path+(i,)) for i,c in enumerate(tree.children)]
        out=a+b if tree.op=="add" else a-b if tree.op=="sub" else a*b
    if cache is not None:cache[path]=out
    return out


def context(tree,path,x):
    cache={}
    evaluate(tree,x,cache)
    derivative=np.ones_like(x)/x.shape[1]
    cursor=()
    node=tree
    for i in path:
        if node.op=="sub" and i==1:derivative=-derivative
        elif node.op=="mul":derivative=derivative*cache[cursor+(1-i,)]
        cursor+=(i,)
        node=node.children[i]
    return cache[path],derivative


def search(x,y,method,seed,cfg):
    start=perf_counter()
    init_rng=np.random.default_rng(np.random.SeedSequence([seed,711]))
    rng=np.random.default_rng(np.random.SeedSequence([seed,712]))
    counts=dict(fitness_evaluations=0,semantic_probes=0,context_passes=0,
                node_sample_evaluations=0,offspring=0,duplicate_offspring=0)
    def used():return sum(counts[k] for k in ("fitness_evaluations","semantic_probes","context_passes"))
    def charge(kind,tree):
        if used()>=cfg.budget:raise RuntimeError("Search budget overrun")
        counts[kind]+=1
        counts["node_sample_evaluations"]+=tree.size*x.size
    def assess(tree):
        charge("fitness_evaluations",tree)
        pred=evaluate(tree,x).mean(axis=1)
        mse=float(np.mean((pred-y)**2))
        return dict(tree=tree,pred=pred,mse=mse,score=mse+cfg.parsimony*tree.size)
    trees=[]
    while len(trees)<cfg.population:
        tree=random_tree(init_rng,int(init_rng.integers(1,cfg.init_depth+1)))
        if valid(tree,cfg):trees.append(tree)
    initial=hashlib.sha256(json.dumps([t.to_dict() for t in trees],sort_keys=True).encode()).hexdigest()
    population=[assess(t) for t in trees]
    best=min(population,key=lambda p:p["score"])
    def tournament():return min((population[int(i)] for i in rng.integers(len(population),size=cfg.tournament)),key=lambda p:p["score"])
    def donor_proposal(parent,path,donor):
        paths=[p for p in donor.paths() if valid(parent.replace(path,donor.at(p)),cfg)]
        return donor.at(paths[int(rng.integers(len(paths)))]) if paths else parent.at(path)
    while used()<cfg.budget:
        parent=tournament()
        tree=parent["tree"]
        paths=list(tree.paths())
        path=paths[int(rng.integers(len(paths)))]
        event=rng.random()
        if event<cfg.crossover_rate:
            donor=tournament()["tree"]
            if method=="tn_dr" and cfg.budget-used()>=3:
                charge("context_passes",tree)
                s,j=context(tree,path,x)
                residual=y-parent["pred"]
                chosen=tree.at(path)
                best_score=-np.inf
                for _ in range(min(cfg.guide_candidates,cfg.budget-used()-1)):
                    candidate=donor_proposal(tree,path,donor)
                    charge("semantic_probes",candidate)
                    delta=np.sum(j*(evaluate(candidate,x)-s),axis=1)
                    score=float(np.mean(2*residual*delta-delta**2))
                    if score>best_score:chosen,best_score=candidate,score
                child=tree.replace(path,chosen)
            else:child=tree.replace(path,donor_proposal(tree,path,donor))
        elif event<cfg.crossover_rate+cfg.mutation_rate:
            child=tree.replace(path,random_tree(rng,2))
            if not valid(child,cfg):child=tree
        else:child=tree
        offspring=assess(child)
        counts["offspring"]+=1
        counts["duplicate_offspring"]+=int(child==tree)
        worst=max(range(len(population)),key=lambda i:population[i]["score"])
        if offspring["score"]<population[worst]["score"]:population[worst]=offspring
        if offspring["score"]<best["score"]:best=offspring
    coefficients=polynomial(best["tree"])
    degrees=[int(k) for k in range(1,len(coefficients)) if abs(coefficients[k])>1e-10]
    return dict(tree=best["tree"].to_dict(),expression=str(best["tree"]),
                coefficients=coefficients.tolist(),degrees=degrees,search_train_mse=best["mse"],
                initial_population_sha256=initial,budget_used=used(),config=asdict(cfg),
                search_seconds=perf_counter()-start,**counts)
