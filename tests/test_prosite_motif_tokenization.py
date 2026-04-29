from pathlib import Path

from cliper.data import (
    build_motif_special_tokens,
    load_motif_specs,
    parse_prosite_dat,
    tokenize_sequence_with_motifs,
)


def test_parse_prosite_dat_keeps_only_pattern_entries(tmp_path: Path) -> None:
    prosite = tmp_path / "prosite.dat"
    prosite.write_text(
        """ID   ASN_GLYCOSYLATION; PATTERN.
AC   PS00001;
DE   N-glycosylation site.
PA   N-{P}-[ST]-{P}.
//
ID   IGNORED_PROFILE; MATRIX.
AC   PS99999;
DE   Not a pattern.
//
""",
        encoding="utf-8",
    )

    motifs = parse_prosite_dat(prosite)

    assert [motif["id"] for motif in motifs] == ["ASN_GLYCOSYLATION"]
    assert motifs[0]["ac"] == "PS00001"
    assert motifs[0]["pa"] == "N-{P}-[ST]-{P}"
    assert motifs[0]["token"] == "<PROSITE:PS00001>"


def test_prosite_span_replacement_is_longest_non_overlapping(tmp_path: Path) -> None:
    motif_json = tmp_path / "motifs.json"
    motif_json.write_text(
        """{
  "motifs": [
    {"id": "SHORT", "kind": "prosite", "ac": "PS00001", "pa": "A-C."},
    {"id": "LONG", "kind": "prosite", "ac": "PS00002", "pa": "A-C-D."}
  ]
}
""",
        encoding="utf-8",
    )
    specs = load_motif_specs(motif_json, matching="prosite")

    tokens, spans, matches = tokenize_sequence_with_motifs("MACDG", specs)

    assert build_motif_special_tokens(specs) == ["<PROSITE:PS00001>", "<PROSITE:PS00002>"]
    assert tokens == ["M", "<PROSITE:PS00002>", "G"]
    assert spans == [1, 3, 1]
    assert [(match.motif_id, match.start, match.end) for match in matches] == [("LONG", 1, 4)]
