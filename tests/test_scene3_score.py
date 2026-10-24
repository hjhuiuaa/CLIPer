from cliper.scene3 import max_window_mean, median_linker_run_length, protein_auroc, protein_has_linker


def test_window_mean_picks_the_high_segment():
    probs = [0.0, 0.0, 1.0, 1.0, 0.0]
    assert max_window_mean(probs, 2) == 1.0
    assert protein_has_linker("00220") == 1
    assert protein_has_linker("00110") == 0


def test_median_run_length_and_protein_auroc():
    assert median_linker_run_length(["0022200", "0002"]) == 2
    rows = [
        {"label": 1, "score": 0.9},
        {"label": 1, "score": 0.8},
        {"label": 0, "score": 0.1},
        {"label": 0, "score": 0.2},
    ]
    assert protein_auroc(rows) == 1.0
