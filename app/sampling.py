"""采样覆盖率审计：周期应采时隙 + 容差内一对一匹配（纯计算，只读）。

周期先**量化为整数微秒**（``period_us``）：时隙只能落在微秒网格上，
``start + i·period`` 一律用整数微秒做时间运算，杜绝浮点秒逐次累加造成的
重复时隙。随后**先按查询窗口算时隙数量**再按时隙数生成时刻：

- ``n = (end − start) // period + 1``，``t_i = start + i·period_us``；
- 周期大于窗口（``(end − start) < period``）时 ``n == 1``，只生成起点时隙，
  不再试算下一个（极大周期）时刻，避免越界时间运算引发溢出。

应采时隙数硬上限 :data:`MAX_EXPECTED_SLOTS`（100000）。

以查询起点为**首个应采时刻**，生成的应采时隙均**不晚于终点**
（``t_i ≤ end``；起点不晚于终点时至少一个时隙）。再取**容差扩展范围**
``[start − tolerance, 末时隙 + tolerance]`` 内的已合并读数，按时间顺序做
**一对一**贪心匹配：

- 读数可覆盖时隙 ``t_i`` 当且仅当 ``|读数时刻 − t_i| ≤ tolerance``
  （**含等于**）；
- 每条读数最多覆盖一个时隙，每个时隙最多接收一条读数；
- 每条读数优先匹配其容差窗口内**最早**（时刻最小）的未覆盖时隙。

连续的缺测时隙合并为缺口，返回缺口起止时隙时刻与缺测时隙数。
全部计算确定性：相同输入恒得相同输出；不修改任何输入。
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Protocol

#: 应采时隙数硬上限：查询窗口 / 量化周期 + 1 超过该值即判参数非法（422）
MAX_EXPECTED_SLOTS = 100_000

#: ``datetime.timedelta`` 可表示的最大微秒数（约 999999999 天）
MAX_DELTA_MICROSECONDS = 10**6 * 86_400 * 999_999_999


class SamplingParamsError(ValueError):
    """采样覆盖率审计入参非法（由接口层转为 422）。"""

    def __init__(self, message: str, *, parameter: str,
                 limit: int | None = None, actual: int | None = None):
        super().__init__(message)
        self.message = message
        self.parameter = parameter
        self.limit = limit
        self.actual = actual


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


def normalize_period(period_seconds: float) -> int:
    """把浮点秒周期量化为**可表示的整数微秒**。

    时隙时刻只能落在微秒网格上，故周期按四舍五入取整到微秒（与
    ``timedelta(seconds=...)`` 内部的微秒化规则一致）。量化后不足 1 微秒
    （含非有限值）一律非法，由接口层返回 422。
    """
    if not isinstance(period_seconds, (int, float)) or not math.isfinite(
        period_seconds
    ):
        raise SamplingParamsError(
            "period_seconds 必须为有限正数",
            parameter="period_seconds",
        )
    period_us = round(period_seconds * 1_000_000)
    if period_us < 1:
        raise SamplingParamsError(
            "period_seconds 量化到整数微秒后不足 1 微秒",
            parameter="period_seconds",
            limit=1,
            actual=period_us,
        )
    return period_us


def make_tolerance(tolerance_seconds: float) -> dt.timedelta:
    """把匹配容差量化为整数微秒的 ``timedelta``（含等于语义不变）。

    容差非负、严格小于周期由接口层校验。容差本身可能远大于
    ``timedelta`` 的可表示范围（如极大周期配大容差）：此时截断到
    :data:`MAX_DELTA_MICROSECONDS`——该量级远超任何合法 datetime 之间的
    距离，截断对匹配结果无影响，且避免构造增量时溢出。
    """
    tolerance_us = max(0, round(tolerance_seconds * 1_000_000))
    return dt.timedelta(
        microseconds=min(tolerance_us, MAX_DELTA_MICROSECONDS)
    )


def plan_slot_count(
    start: dt.datetime, end: dt.datetime, period_us: int
) -> int:
    """按查询窗口计算应采时隙数（先算数、不生成时刻）。

    ``n = floor((end − start) / period) + 1``。窗口短于一个周期时只有起点
    时隙（``n == 1``），全程仅做整数微秒运算，不会试算下一个（可能越界的）
    时隙时刻。超过 :data:`MAX_EXPECTED_SLOTS` 抛 :class:`SamplingParamsError`。
    """
    window_us = int((end - start) // dt.timedelta(microseconds=1))
    count = window_us // period_us + 1
    if count > MAX_EXPECTED_SLOTS:
        raise SamplingParamsError(
            f"应采时隙数超过上限 {MAX_EXPECTED_SLOTS}",
            parameter="period_seconds",
            limit=MAX_EXPECTED_SLOTS,
            actual=count,
        )
    return count


def generate_slots(
    start: dt.datetime, end: dt.datetime, period_us: int
) -> list[dt.datetime]:
    """以 start 为首个应采时刻，按整数微秒周期生成不晚于 end 的时隙（升序）。

    时隙数先经 :func:`plan_slot_count` 由查询窗口算出；数量为 1（周期大于
    窗口，含极大周期）时直接返回起点，不构造下一个时隙的时间增量。
    """
    count = plan_slot_count(start, end, period_us)
    if count == 1:  # 周期 ≥ 窗口：只取起点，避免极大 period_us 的越界运算
        return [start]
    step = dt.timedelta(microseconds=period_us)
    return [start + step * i for i in range(count)]


def _sub_saturating(t: dt.datetime, delta: dt.timedelta) -> dt.datetime:
    """``t − delta``，结果早于 :data:`datetime.min` 时饱和截断到 min。"""
    try:
        return t - delta
    except OverflowError:
        return dt.datetime.min.replace(tzinfo=t.tzinfo)


def _add_saturating(t: dt.datetime, delta: dt.timedelta) -> dt.datetime:
    """``t + delta``，结果晚于 :data:`datetime.max` 时饱和截断到 max。"""
    try:
        return t + delta
    except OverflowError:
        return dt.datetime.max.replace(tzinfo=t.tzinfo)


def compute_sampling_coverage(
    samples: list[_SampleLike],
    *,
    start: dt.datetime,
    end: dt.datetime,
    period_us: int,
    tolerance: dt.timedelta,
) -> SamplingCoverageResult:
    """周期时隙的采样覆盖率与连续缺测缺口（只读纯计算）。

    参数合法性（``start ≤ end``、``period_us ≥ 1``、应采时隙数不超过
    :data:`MAX_EXPECTED_SLOTS`、``0 ≤ tolerance < 周期``）由接口层校验，
    本函数假定入参合法。samples 无需预先排序、无需预先按范围过滤；
    只读取容差扩展范围内的读数，范围外读数不参与匹配。
    """
    # 唯一生效时隙序列：匹配、覆盖率、缺口均基于同一序列
    slots = generate_slots(start, end, period_us)
    expected = len(slots)
    if expected == 0:  # 防御：接口层已保证 start ≤ end（至少一个时隙）
        return SamplingCoverageResult(
            expected_count=0, covered_count=0, coverage=0.0, gaps=[]
        )

    # 容差扩展范围：覆盖全部时隙匹配窗口的最小区间。
    # 容差（或单时隙配大周期场景）可能使边界越出 datetime 可表示范围，
    # 此时饱和截断到 min/max：任何真实读数都在截断后的区间内。
    lo = _sub_saturating(slots[0], tolerance)
    hi = _add_saturating(slots[-1], tolerance)
    readings = sorted(s.ts for s in samples if lo <= s.ts <= hi)

    covered = [False] * expected
    covered_count = 0
    for ts in readings:
        # 该读数可覆盖的时隙下标区间 [left, right)；
        # 比较边界同样做饱和截断，保证大容差下时间运算不越界
        left = bisect_left(slots, _sub_saturating(ts, tolerance))
        right = bisect_right(slots, _add_saturating(ts, tolerance))
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
