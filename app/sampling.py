"""采样覆盖率审计：周期应采时隙 + 容差内一对一匹配（纯计算，只读）。

以查询起点为**首个应采时刻**，按固定周期生成**不晚于终点**的应采时隙
（``t_i = start + i·period``，``t_i ≤ end``；起点不晚于终点时至少一个
时隙）。再取**容差扩展范围** ``[start − tolerance, 末时隙 + tolerance]``
内的已合并读数，按时间顺序做**一对一**贪心匹配：

- 读数可覆盖时隙 ``t_i`` 当且仅当 ``|读数时刻 − t_i| ≤ tolerance``
  （**含等于**）；
- 每条读数最多覆盖一个时隙，每个时隙最多接收一条读数；
- 每条读数优先匹配其容差窗口内**最早**（时刻最小）的未覆盖时隙。

连续的缺测时隙合并为缺口，返回缺口起止时隙时刻与缺测时隙数。
全部计算确定性：相同输入恒得相同输出；不修改任何输入。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Protocol


class _SampleLike(Protocol):
    ts: dt.datetime


@dataclass(frozen=True)
class CoverageGap:
    """连续缺测时隙合并得到的缺口。"""

    start_ts: dt.datetime  # 缺口内首个缺测时隙的时刻
    end_ts: dt.datetime  # 缺口内末个缺测时隙的时刻
    missing_slots: int  # 缺口内连续缺测的时隙数


@dataclass(frozen=True)
class SamplingCoverageResult:
    expected_count: int  # 应采时隙数
    covered_count: int  # 已覆盖时隙数
    coverage: float  # 覆盖率 = covered / expected（expected 为 0 时取 0.0）
    gaps: list[CoverageGap]


def generate_slots(
    start: dt.datetime, end: dt.datetime, period_seconds: float
) -> list[dt.datetime]:
    """以 start 为首个应采时刻，按周期生成不晚于 end 的时隙（升序）。"""
    slots: list[dt.datetime] = []
    i = 0
    while True:
        ts = start + dt.timedelta(seconds=period_seconds * i)
        if ts > end:
            break
        slots.append(ts)
        i += 1
    return slots


def compute_sampling_coverage(
    samples: list[_SampleLike],
    *,
    start: dt.datetime,
    end: dt.datetime,
    period_seconds: float,
    tolerance_seconds: float,
) -> SamplingCoverageResult:
    """周期时隙的采样覆盖率与连续缺测缺口（只读纯计算）。

    参数合法性（``start ≤ end``、``period_seconds > 0``、
    ``0 ≤ tolerance_seconds < period_seconds``）由接口层校验，
    本函数假定入参合法。samples 无需预先排序、无需预先按范围过滤；
    只读取容差扩展范围内的读数，范围外读数不参与匹配。
    """
    slots = generate_slots(start, end, period_seconds)
    expected = len(slots)
    if expected == 0:  # 防御：接口层已保证 start ≤ end（至少一个时隙）
        return SamplingCoverageResult(
            expected_count=0, covered_count=0, coverage=0.0, gaps=[]
        )

    tolerance = dt.timedelta(seconds=tolerance_seconds)
    # 容差扩展范围：覆盖全部时隙匹配窗口的最小区间
    lo = slots[0] - tolerance
    hi = slots[-1] + tolerance
    readings = sorted(s.ts for s in samples if lo <= s.ts <= hi)

    covered = [False] * expected
    covered_count = 0
    for ts in readings:
        # 该读数可覆盖的时隙下标区间 [left, right)
        left = bisect_left(slots, ts - tolerance)
        right = bisect_right(slots, ts + tolerance)
        for i in range(left, right):  # 优先最早的可覆盖时隙
            if not covered[i]:
                covered[i] = True
                covered_count += 1
                break

    gaps: list[CoverageGap] = []
    i = 0
    while i < expected:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < expected and not covered[j + 1]:
            j += 1
        gaps.append(
            CoverageGap(
                start_ts=slots[i], end_ts=slots[j], missing_slots=j - i + 1
            )
        )
        i = j + 1

    return SamplingCoverageResult(
        expected_count=expected,
        covered_count=covered_count,
        coverage=covered_count / expected,
        gaps=gaps,
    )
