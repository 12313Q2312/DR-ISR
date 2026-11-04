import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dr_isr.tree import Node
from neural.models import PolynomialNeuron,build_model
from neural.vector_gp import SearchConfig,context,evaluate,polynomial,search
from neural.experiment import load_split,train_model

torch.set_num_threads(2)


class NeuralTests(unittest.TestCase):
    def test_polynomial_neuron_matches_author_parameterization_and_gradient(self):
        layer=PolynomialNeuron(2,1,[1,2]).double()
        with torch.no_grad():
            layer.weights[0].copy_(torch.tensor([[2.,3.]]))
            layer.weights[1].copy_(torch.tensor([[4.,5.]]))
            layer.bias.fill_(.5)
        x=torch.tensor([[.2,-.4],[.6,.8]],dtype=torch.float64,requires_grad=True)
        expected=2*x[:,0]+3*x[:,1]+4*x[:,0]**2+5*x[:,1]**2+.5
        torch.testing.assert_close(layer(x).flatten(),expected)
        self.assertTrue(torch.autograd.gradcheck(layer,(x,)))

    def test_models_ceiling_and_finite_gradients(self):
        for method in ("tn_vsr","tn_dr","kan","mlp"):
            torch.manual_seed(3)
            model,width,count=build_model(method,10,[1,2,4])
            self.assertLessEqual(count,256)
            x=torch.randn(8,10)*.3
            loss=model(x).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if method=="kan":
                bases=model.layers[0].b_splines(x)
                torch.testing.assert_close(bases.sum(dim=-1),torch.ones_like(x),atol=1e-5,rtol=1e-5)

    def test_polynomial_expansion_context_and_guided_budget(self):
        x=Node("x",value=0)
        tree=Node("mul",(Node("add",(x,Node("const",value=1))),x))
        data=np.array([[.2,.3],[-.5,.7],[.9,-.9]])
        np.testing.assert_allclose(polynomial(tree),[0,1,1])
        np.testing.assert_allclose(evaluate(tree,data),data+data**2)
        for path in tree.paths():
            s,j=context(tree,path,data)
            h=1e-5
            old=tree.at(path)
            plus=tree.replace(path,Node("add",(old,Node("const",value=h))))
            minus=tree.replace(path,Node("sub",(old,Node("const",value=h))))
            numeric=(evaluate(plus,data).mean(axis=1)-evaluate(minus,data).mean(axis=1))/(2*h)
            np.testing.assert_allclose(j.sum(axis=1),numeric,atol=1e-8)
        rng=np.random.default_rng(2)
        data=rng.uniform(-1,1,(16,3))
        y=(data**2).mean(axis=1)
        for budget in (16,17,18,19,80):
            cfg=SearchConfig(budget=budget,population=16)
            a=search(data,y,"tn_vsr",3,cfg)
            b=search(data,y,"tn_dr",3,cfg)
            self.assertEqual(a["initial_population_sha256"],b["initial_population_sha256"])
            for row in (a,b):
                self.assertEqual(row["budget_used"],budget)
                self.assertEqual(budget,sum(row[k] for k in ("fitness_evaluations","semantic_probes","context_passes")))
                self.assertGreater(len(row["degrees"]),0)
                prediction=evaluate(Node.from_dict(row["tree"]),data).mean(axis=1)
                self.assertAlmostEqual(row["search_train_mse"],float(np.mean((prediction-y)**2)))

    def test_scaling_excludes_validation_and_test(self):
        cfg=dict(split_seed=19,train_fraction=.6,validation_fraction=.2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"epsilon_low_no1.csv"
            rng=np.random.default_rng(1)
            raw=rng.normal(size=(30,4))
            np.savetxt(path,raw,delimiter=",",header="x0,x1,x2,y",comments="")
            a,meta=load_split(tmp,"epsilon_low","no1",cfg)
            raw[meta["validation_ids"]+meta["test_ids"]]=1e8
            np.savetxt(path,raw,delimiter=",",header="x0,x1,x2,y",comments="")
            b,after=load_split(tmp,"epsilon_low","no1",cfg)
            np.testing.assert_array_equal(a["train"][0],b["train"][0])
            np.testing.assert_array_equal(a["train"][1],b["train"][1])
            self.assertEqual(meta["target_mean"],after["target_mean"])
            self.assertEqual(meta["target_std"],after["target_std"])

    def test_checkpoint_selection_and_holdout_isolation(self):
        cfg=dict(training_steps=10,checkpoint_every=2,learning_rate=.01)
        rng=np.random.default_rng(6)
        split={name:(rng.normal(size=(12,2)),rng.normal(size=12)) for name in ("train","validation","test")}
        torch.manual_seed(2)
        model,_,_=build_model("tn_vsr",2,[1,2],64)
        other=copy.deepcopy(model)
        a=train_model(model,split,cfg)
        changed={**split,"test":(split["test"][0],np.full(12,1e6))}
        b=train_model(other,changed,cfg)
        self.assertEqual(a["best_validation_step"],b["best_validation_step"])
        for k,v in model.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k])
        self.assertAlmostEqual(a["validation_nmse"],min(h["validation_nmse"] for h in a["history"]),places=6)
        self.assertEqual(a["optimizer_updates"],10)


if __name__=="__main__":unittest.main()
