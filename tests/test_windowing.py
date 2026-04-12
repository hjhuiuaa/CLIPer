from cliper.windowing import build_eval_window_starts, training_crop


def test_training_crop_contains_positive_region() -> None:
    sequence = "A" * 300
    labels = "0" * 120 + "1" * 30 + "0" * 150
    cropped_seq, cropped_labels, start = training_crop(sequence, labels, window_size=128, seed=7)
    assert len(cropped_seq) == 128
    assert len(cropped_labels) == 128
    assert "1" in cropped_labels
    assert 0 <= start <= len(sequence) - 128


def test_eval_windows_are_label_agnostic_and_cover_sequence() -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 20  # length 400
    starts = build_eval_window_starts(sequence, window_size=128, stride=64, top_k_heuristic=3)
    assert starts[0] == 0
    assert starts[-1] == len(sequence) - 128
    assert len(starts) >= 5

