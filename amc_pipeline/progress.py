from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    enabled: bool = True,
    leave: bool = False,
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True, leave=leave)
    except Exception:
        return _basic_progress(iterable, desc=desc, total=total, unit=unit)


def _basic_progress(iterable: Iterable[T], *, desc: str, total: int | None, unit: str) -> Iterator[T]:
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    count = 0
    next_report = 0
    for item in iterable:
        count += 1
        if total:
            percent = int((count / total) * 100)
            if count == 1 or count == total or percent >= next_report:
                print(f"{desc}: {count}/{total} {unit}", file=sys.stderr)
                next_report = min(100, percent + 10)
        elif count == 1 or count % 100 == 0:
            print(f"{desc}: {count} {unit}", file=sys.stderr)
        yield item
