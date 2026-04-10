import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.time import Time

import padre_meddea.util.meta as meta


def test_summarize_files_handles_missing_sizes():
    padre_results = Table(
        rows=[
            (Time("2025-01-10T00:00:00"), 10),
            (Time("2025-01-15T00:00:00"), None),
            (Time("2025-02-01T00:00:00"), 30),
        ],
        names=("Time", "File Size"),
    )
    results = {"padre": padre_results}

    total_files, total_size = meta.summarize_files(results)

    assert total_files == 3
    assert total_size == 40


def test_bin_files_by_week_groups_counts_sizes_and_week_number():
    padre_results = Table(
        rows=[
            (Time("2025-01-10T00:00:00"), 10),
            (Time("2025-01-15T00:00:00"), None),
            (Time("2025-02-01T00:00:00"), 30),
            ("not-a-time", 50),
        ],
        names=("Time", "File Size"),
    )
    results = {"padre": padre_results}

    summary = meta.bin_files_by_week(results)

    assert list(summary.time.strftime("%Y-%m-%d")) == [
        "2025-01-06",
        "2025-01-13",
        "2025-01-27",
    ]
    assert list(summary["Week Number"]) == [2, 3, 5]
    assert list(summary["Total Files"]) == [1, 1, 1]
    assert summary["Total Size"].unit == u.Mbyte
    np.testing.assert_allclose(
        summary["Total Size"].to_value(u.Mbyte),
        u.Quantity([10, 0, 30], u.byte).to_value(u.Mbyte),
    )


def test_queryresponse_to_timeseries_uses_file_time_as_index():
    padre_results = Table(
        rows=[
            (Time("2025-02-01T00:00:00"), "b.fits", 20),
            (Time("2025-01-01T00:00:00"), "a.fits", 10),
        ],
        names=("Time", "File Name", "File Size"),
    )
    results = {"padre": padre_results}

    ts = meta.queryresponse_to_timeseries(results)

    assert list(ts.time.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-02-01"]
    assert list(ts["File Name"]) == ["a.fits", "b.fits"]
    assert ts["File Size"].unit == u.byte
    assert list(ts["File Size"].to_value(u.byte)) == [10, 20]


def test_queryresponse_to_timeseries_skips_invalid_times():
    padre_results = Table(
        rows=[
            ("not-a-time", "bad.fits", 999),
            (Time("2025-01-01T00:00:00"), "good.fits", 10),
        ],
        names=("Time", "File Name", "File Size"),
    )
    results = {"padre": padre_results}

    ts = meta.queryresponse_to_timeseries(results)

    assert len(ts) == 1
    assert list(ts.time.strftime("%Y-%m-%d")) == ["2025-01-01"]
    assert list(ts["File Name"]) == ["good.fits"]


def test_queryresponse_to_timeseries_preserves_all_non_time_columns():
    padre_results = Table(
        rows=[
            (
                Time("2025-01-01T00:00:00"),
                "a.fits",
                ".fits",
                "l1",
                "spectrum",
                10,
                "https://example.org/a.fits",
            ),
            (
                Time("2025-01-02T00:00:00"),
                "b.dat",
                ".dat",
                "raw",
                "housekeeping",
                None,
                "https://example.org/b.dat",
            ),
        ],
        names=(
            "Time",
            "File Name",
            "File Extension",
            "Level",
            "Descriptor",
            "File Size",
            "url",
        ),
    )
    results = {"padre": padre_results}

    ts = meta.queryresponse_to_timeseries(results)

    expected_cols = {
        "File Name",
        "File Extension",
        "Level",
        "Descriptor",
        "File Size",
        "url",
    }
    assert expected_cols.issubset(set(ts.colnames))
    assert ts["File Size"].unit == u.byte


def test_file_weekly_summary_rolls_up_queryresponse_output():
    padre_results = Table(
        rows=[
            (Time("2025-01-10T00:00:00"), "a.fits", 10),
            (Time("2025-01-15T00:00:00"), "b.fits", 20),
            (Time("2025-01-16T00:00:00"), "c.fits", 30),
        ],
        names=("Time", "File Name", "File Size"),
    )
    results = {"padre": padre_results}

    file_ts = meta.queryresponse_to_timeseries(results)
    weekly_ts = meta.file_weekly_summary(file_ts)

    assert list(weekly_ts.time.strftime("%Y-%m-%d")) == ["2025-01-06", "2025-01-13"]
    assert list(weekly_ts["Week Number"]) == [2, 3]
    assert list(weekly_ts["Total Files"]) == [1, 2]
    assert weekly_ts["Total Size"].unit == u.Mbyte
    np.testing.assert_allclose(
        weekly_ts["Total Size"].to_value(u.Mbyte),
        u.Quantity([10, 50], u.byte).to_value(u.Mbyte),
    )
