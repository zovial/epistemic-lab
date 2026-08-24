from epistemic_lab.cfr import CFRSolver, exploitability


def test_cfr_builds_average_strategy_with_finite_exploitability():
    solver = CFRSolver().train(5)
    avg = solver.average_strategy()
    assert len(avg) > 0
    assert exploitability(avg) > 0.0
