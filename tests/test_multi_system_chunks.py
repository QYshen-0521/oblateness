"""分块与合并工具（578 行 / 20 组）。"""
import sys
from pathlib import Path

import oblateness.merge_multi_system_chunks as merge_mod
from oblateness.merge_multi_system_chunks import (
    apply_tierb_global_row_index_offset,
    merge_summaries,
)
from oblateness.multi_system_chunks import partition_indices, rows_csv_for_chunk


def test_partition_578_20():
    chunks = partition_indices(578, 20)
    assert len(chunks) == 20
    assert sum(len(c) for c in chunks) == 578
    assert chunks[0][0] == 0 and chunks[-1][-1] == 577
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1
    assert sizes.count(29) == 18 and sizes.count(28) == 2


def test_rows_csv_roundtrip():
    s = rows_csv_for_chunk(3, 20, 578)
    assert s.startswith("87,") and s.endswith("115")


def test_merge_summaries_dedupe(tmp_path):
    c0 = tmp_path / "chunk_00"
    c1 = tmp_path / "chunk_01"
    c0.mkdir()
    c1.mkdir()
    (c0 / "multi_system_summary.csv").write_text(
        "row_index,pl_name,n_success,n_failed,figure_path,npz_path,error\n"
        "0,A,1,0,p,,,\n",
        encoding="utf-8",
    )
    (c1 / "multi_system_summary.csv").write_text(
        "row_index,pl_name,n_success,n_failed,figure_path,npz_path,error\n"
        "1,B,2,0,q,,,\n",
        encoding="utf-8",
    )
    merged, warns = merge_summaries(tmp_path)
    assert len(merged) == 2
    assert not warns
    assert merged[0]["n_success"] == "1" and merged[0]["n_failed"] == "0"


def test_merge_summaries_normalizes_legacy_three_column_counts(tmp_path):
    c0 = tmp_path / "chunk_00"
    c0.mkdir()
    (c0 / "multi_system_summary.csv").write_text(
        "row_index,pl_name,n_clean,n_degenerate,n_failed,figure_path,npz_path,error\n"
        "3,X,10,5,2,p,,,\n",
        encoding="utf-8",
    )
    merged, warns = merge_summaries(tmp_path)
    assert len(merged) == 1
    assert merged[0]["n_success"] == "15"
    assert merged[0]["n_failed"] == "2"


def test_main_appends_to_existing_multi_system_csv(tmp_path, monkeypatch):
    (tmp_path / "results" / "multi_system_a1_only").mkdir(parents=True)
    out_csv = tmp_path / "results" / "multi_system_a1_only" / "multi_system_summary_a1only.csv"
    out_csv.write_text(
        "row_index,pl_name,n_success,n_failed,figure_path,npz_path,error\n"
        "99,Z,1,0,p,,,\n",
        encoding="utf-8",
    )
    c0 = tmp_path / "results" / "multi_system_a1_only" / "_chunks_tierb" / "chunk_00"
    c0.mkdir(parents=True)
    (c0 / "multi_system_summary.csv").write_text(
        "row_index,pl_name,n_success,n_failed,figure_path,npz_path,error\n"
        "0,A,1,0,q,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(merge_mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", [""])
    merge_mod.main()
    text = out_csv.read_text(encoding="utf-8")
    assert "99,Z" in text
    assert "578,A" in text
    assert "0,A" not in text
    assert text.count("row_index,pl_name") == 1


def test_apply_tierb_row_index_offset_only_when_small_csv(monkeypatch):
    import oblateness.merge_multi_system_chunks as m

    p = Path("_chunks_tierb")
    monkeypatch.setattr("oblateness.multi_system_chunks.csv_data_row_count", lambda: 164)
    assert m.should_apply_tierb_row_index_offset(p) is True

    monkeypatch.setattr("oblateness.multi_system_chunks.csv_data_row_count", lambda: 742)
    assert m.should_apply_tierb_row_index_offset(p) is False

    monkeypatch.setattr("oblateness.multi_system_chunks.csv_data_row_count", lambda: 578)
    assert m.should_apply_tierb_row_index_offset(p) is False

    assert m.should_apply_tierb_row_index_offset(Path("_chunks")) is False
