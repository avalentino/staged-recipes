import pytest

from eopf.dask_utils.dask_helpers import (
    estimate_max_workers,
    estimate_maximal_chunk_size,
    estimate_min_array_size_mb,
    estimate_min_chunk_size_mb,
)


def test_estimate_maximal_chunk_size() -> None:
    assert estimate_maximal_chunk_size(1000, 4) == pytest.approx(30.0)


def test_estimate_min_chunk_size_mb() -> None:
    assert estimate_min_chunk_size_mb(1000, 4) == pytest.approx(100.0)
    assert estimate_min_chunk_size_mb(20000, 1) == pytest.approx(240.0)


def test_estimate_min_array_size_mb() -> None:
    assert estimate_min_array_size_mb(2, 3) == 6000


def test_estimate_max_workers_without_cluster_cap() -> None:
    assert estimate_max_workers(10000, 2, 100) == 7


def test_estimate_max_workers_with_cluster_cap_and_worker_cap() -> None:
    assert (
        estimate_max_workers(
            10000,
            2,
            100,
            cluster_memory_limit_mb=20000,
            n_workers=10,
        )
        == 10
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"memory_limit": 0, "threads_per_worker": 1, "chunk_size_mb": 1},
            "memory_limit must be strictly positive",
        ),
        (
            {"memory_limit": 1, "threads_per_worker": 0, "chunk_size_mb": 1},
            "threads_per_worker must be strictly positive",
        ),
        (
            {"memory_limit": 1, "threads_per_worker": 1, "chunk_size_mb": 0},
            "chunk_size_mb must be strictly positive",
        ),
        (
            {
                "memory_limit": 1,
                "threads_per_worker": 1,
                "chunk_size_mb": 1,
                "n_workers": 0,
            },
            "n_workers must be strictly positive",
        ),
    ],
)
def test_estimate_max_workers_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_max_workers(**kwargs)
