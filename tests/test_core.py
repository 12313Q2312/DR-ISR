import unittest
import numpy as np
from dr_isr.data import make_dataset
from dr_isr.gp import Config, Ledger, METHODS, residual_score, search
from dr_isr.tree import EPS, LIMIT, Node, context, evaluate

X = Node("x", value=0)
C = lambda n: Node("const", value=n)


class TreeTests(unittest.TestCase):
    def test_composition_serialization_and_immutable_replacement(self):
        inner = Node("mul", (X, X))
        tree = Node("sin", (inner,))
        data = np.array([[-1.], [0.], [2.]])
        np.testing.assert_allclose(evaluate(tree, data), np.sin(data[:,0] ** 2))
        self.assertEqual(Node.from_dict(tree.to_dict()), tree)
        self.assertEqual(tree.replace((0,), X), Node("sin", (X,)))
        self.assertEqual(tree.at((0,)), inner)
        self.assertEqual(tree.size, 4)
        self.assertEqual(tree.depth, 2)

    def test_guards(self):
        data = np.array([[0.], [EPS / 2], [1.]])
        pred = evaluate(Node("div", (C(2), X)), data)
        np.testing.assert_allclose(pred, [1, 1, 2])
        np.testing.assert_allclose(evaluate(Node("mul", (C(LIMIT), C(LIMIT))), data), LIMIT)
        _, derivative = context(Node("div", (C(2), X)), (1,), data)
        np.testing.assert_allclose(derivative, [0, 0, -2])
        _, derivative = context(Node("mul", (C(LIMIT), C(LIMIT))), (0,), data)
        np.testing.assert_allclose(derivative, 0)

    def test_context_sensitivity_matches_finite_differences(self):
        # All six operators, both child positions, and a nested nonlinear context.
        data = np.array([[-.7], [.2], [.9]])
        trees = [Node(op, (X, C(.7))) for op in ("add", "sub", "mul", "div")]
        trees += [Node("sin", (X,)), Node("cos", (X,)),
                  Node("sin", (Node("mul", (X, X)),))]
        h = 1e-6
        for tree in trees:
            for path in tree.paths():
                s = tree.at(path)
                plus = tree.replace(path, Node("add", (s, C(h))))
                minus = tree.replace(path, Node("sub", (s, C(h))))
                numeric = (evaluate(plus, data) - evaluate(minus, data)) / (2*h)
                _, analytic = context(tree, path, data)
                np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    def test_guidance_matches_exact_mse_change_in_affine_context(self):
        data = np.array([[-1.], [0.], [1.]])
        tree = Node("add", (C(2), X))
        donor = Node("mul", (X, X))
        y = np.array([3., 2., 3.])
        old = evaluate(tree, data)
        s, j = context(tree, (1,), data)
        score = residual_score(y-old, s, evaluate(donor, data), j)
        new = evaluate(tree.replace((1,), donor), data)
        self.assertAlmostEqual(score, np.mean((y-old)**2)-np.mean((y-new)**2))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.data = make_dataset("additive", 99, 20, 24)
        self.config = Config(budget=100, population=8, init_depth=2, trace_every=10)

    def test_determinism_shared_initialization_and_budget(self):
        initial = None
        for method in METHODS:
            a = search(self.data.train_x, self.data.train_y, method, 7, self.config)
            b = search(self.data.train_x, self.data.train_y, method, 7, self.config)
            for field in ("tree", "train_mse", "fitness_evaluations", "semantic_probes", "context_passes"):
                self.assertEqual(a[field], b[field])
            self.assertEqual(a["budget_used"], self.config.budget)
            self.assertEqual(a["budget_used"], a["fitness_evaluations"]+a["semantic_probes"]+a["context_passes"])
            if initial is not None:
                self.assertEqual(initial, a["initial_population"])
            initial = a["initial_population"]
            if method == "gp":
                self.assertEqual(a["semantic_probes"], 0)
            else:
                self.assertGreater(a["semantic_probes"], 0)
                self.assertGreater(a["context_passes"], 0)
            tree = Node.from_dict(a["tree"])
            self.assertLessEqual(tree.depth, self.config.max_depth)
            self.assertLessEqual(tree.size, self.config.max_nodes)
            np.testing.assert_allclose(a["train_mse"], np.mean((self.data.train_y-evaluate(tree,self.data.train_x))**2))
            self.assertEqual(a["trace"][-1]["budget_used"], self.config.budget)
            objectives = [p["best_objective"] for p in a["trace"]]
            self.assertTrue(all(b <= a for a,b in zip(objectives,objectives[1:])))

    def test_tail_budgets_and_zero_crossover(self):
        for budget in (8, 9, 10, 11, 12, 13):
            config = Config(budget=budget, population=8, init_depth=2)
            a = search(self.data.train_x,self.data.train_y,"residual_gp",1,config)
            self.assertEqual(a["budget_used"],budget)
        config = Config(budget=40,population=8,init_depth=2,crossover_rate=0)
        a = search(self.data.train_x,self.data.train_y,"gp",1,config)
        b = search(self.data.train_x,self.data.train_y,"residual_gp",1,config)
        self.assertEqual(a["tree"],b["tree"])
        self.assertEqual(b["semantic_probes"],0)

    def test_holdout_is_not_a_search_input(self):
        a = search(self.data.train_x,self.data.train_y,"residual_gp",3,self.config)
        self.data.test_y[:] = 1e100
        self.data.extrap_y[:] = -1e100
        b = search(self.data.train_x,self.data.train_y,"residual_gp",3,self.config)
        self.assertEqual(a["tree"],b["tree"])

    def test_data_reproducibility_noise_and_disjoint_domains(self):
        a = make_dataset("noisy_additive",99,20,24)
        b = make_dataset("noisy_additive",99,20,24)
        self.assertEqual(a.fingerprint,b.fingerprint)
        self.assertTrue(np.all(np.abs(a.extrap_x) >= 1))
        self.assertTrue(np.all(np.abs(a.test_x) <= 1))
        self.assertFalse(np.any(np.all(a.train_x[:,None,:] == a.test_x[None,:,:],axis=2)))
        expected = np.sin(a.test_x[:,0]) + a.test_x[:,1]**2
        np.testing.assert_allclose(a.test_y,expected)

    def test_validation_and_ledger_overrun(self):
        for kwargs in ({"budget":1},{"crossover_rate":2},{"parsimony":-1},{"init_depth":7}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        ledger = Ledger(1)
        ledger.charge("fitness_evaluations",X,20)
        with self.assertRaises(RuntimeError):
            ledger.charge("semantic_probes",X,20)
        with self.assertRaises(ValueError):
            search(self.data.train_x,self.data.train_y,"missing",0,self.config)


if __name__ == "__main__":
    unittest.main()
