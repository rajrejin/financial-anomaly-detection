from finanomaly.experiments.promotion import (
    Candidate,
    GateConfig,
    Row,
    apply_gate,
    gate_candidate,
    select_best,
)


def cand(name, recall, fa=2.0, ap=0.3, std=0.0, expl=True, contract=True, fam="x"):
    return Candidate(name, fam, 1, recall, fa, ap, std, expl, contract)


def test_select_best_tie_breaks():
    rows = [
        Row(0, 1.0, 2.8, 0.3, 10),
        Row(1, 1.0, 2.5, 0.2, 99),  # same recall, lower FA wins over AP
        Row(2, 0.67, 1.0, 0.9, 1),
        Row(3, 1.0, 2.5, 0.4, 50),  # same recall and FA, higher AP wins
        Row(4, 1.0, 2.5, 0.4, 20),  # same everything, simpler wins
    ]
    assert select_best(rows).config_idx == 4


def test_gate_each_criterion_blocks():
    g = GateConfig(budget_per_year=3.0, fa_tolerance=0.5, max_recall_std=0.2)
    ok = cand("ok", 1.0)
    assert gate_candidate(ok, 1.0, None, g)["passes"]
    assert not gate_candidate(cand("low", 0.67), 1.0, None, g)["passes"]
    assert not gate_candidate(cand("fa", 1.0, fa=3.6), 1.0, None, g)["passes"]
    assert gate_candidate(cand("fa", 1.0, fa=3.5), 1.0, None, g)["passes"]
    assert not gate_candidate(cand("spread", 1.0, std=0.25), 1.0, None, g)["passes"]
    assert not gate_candidate(cand("expl", 1.0, expl=False), 1.0, None, g)["passes"]
    assert not gate_candidate(cand("contract", 1.0, contract=False), 1.0, None, g)["passes"]


def test_champion_ranking_and_production_reference():
    a = cand("a", 1.0, fa=2.0, ap=0.2)
    b = cand("b", 1.0, fa=2.0, ap=0.3)
    c = cand("c", 0.67)
    res = apply_gate([a, b, c], 0.67, None)
    assert res["champion"]["name"] == "b"
    # a production model with recall 1.0 blocks a candidate with 0.67
    prod = cand("prod", 1.0)
    res2 = apply_gate([c], 0.5, prod)
    assert res2["champion"] is None


def test_no_candidate_no_champion():
    assert apply_gate([], 1.0, None)["champion"] is None
