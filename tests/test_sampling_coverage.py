"""采样覆盖率审计：周期时隙 + 容差内一对一匹配（接口集成测试）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import (
    Base,
    CaseState,
    Reading,
    SegmentSnapshot,
    SessionLocal,
    engine,
    init_db,
)
from app.main import app


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _reading(sn, ts, t=20.0, h=50.0):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(minutes, seconds=0):
    total = minutes * 60 + seconds
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"2026-09-01T{hh:02d}:{mm:02d}:{ss:02d}Z"


def _post(client, payload):
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text


def _coverage(client, sn, **params):
    return client.get(f"/api/v1/cases/{sn}/sampling-coverage", params=params)


BASE = {
    "start": "2026-09-01T00:00:00Z",
    "end": "2026-09-01T01:00:00Z",
    "period_seconds": 600,
    "tolerance_seconds": 60,
}


# ---------- 时隙生成与基本覆盖 ----------


def test_full_coverage_no_gaps(client):
    _post(client, [_reading("CASE-S1", _ts(m)) for m in range(0, 61, 10)])
    r = _coverage(client, "CASE-S1", **BASE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-S1"
    assert body["start"] == "2026-09-01T00:00:00Z"
    assert body["end"] == "2026-09-01T01:00:00Z"
    assert body["period_seconds"] == 600.0
    assert body["tolerance_seconds"] == 60.0
    assert body["expected_count"] == 7
    assert body["covered_count"] == 7
    assert body["coverage"] == pytest.approx(1.0)
    assert body["gap_count"] == 0
    assert body["gaps"] == []


def test_slots_from_start_not_beyond_end(client):
    _post(client, [_reading("CASE-S2", _ts(m)) for m in range(0, 11)])
    params = {"start": "2026-09-01T00:00:00Z", "period_seconds": 300,
              "tolerance_seconds": 0}
    # end 恰好落在时隙上：00:00 / 00:05 / 00:10 共 3 个
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:10:00Z").json()
    assert body["expected_count"] == 3
    assert body["covered_count"] == 3
    # end 差 1 秒不到下一个时隙：只有 00:00 / 00:05
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:09:59Z").json()
    assert body["expected_count"] == 2
    # start == end：仅起点一个时隙
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:00:00Z").json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1


# ---------- 容差窗口（含等于） ----------


def test_tolerance_window_inclusive(client):
    # 单个时隙 00:00，容差 60 秒：恰好 ±60s 可覆盖，±61s 不可
    _post(client, [
        _reading("CASE-S3A", "2026-08-31T23:59:00Z"),
        _reading("CASE-S3B", "2026-08-31T23:58:59Z"),
        _reading("CASE-S3C", "2026-09-01T00:01:00Z"),
        _reading("CASE-S3D", "2026-09-01T00:01:01Z"),
    ])
    params = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:00:00Z",
              "period_seconds": 600, "tolerance_seconds": 60}
    assert _coverage(client, "CASE-S3A", **params).json()["covered_count"] == 1
    assert _coverage(client, "CASE-S3B", **params).json()["covered_count"] == 0
    assert _coverage(client, "CASE-S3C", **params).json()["covered_count"] == 1
    assert _coverage(client, "CASE-S3D", **params).json()["covered_count"] == 0


def test_tolerance_extended_range_reaches_outside_window(client):
    # 首时隙之前、末时隙之后容差内的读数仍参与匹配
    _post(client, [
        _reading("CASE-S4", "2026-08-31T23:59:30Z"),
        _reading("CASE-S4", "2026-09-01T00:10:30Z"),
    ])
    body = _coverage(
        client, "CASE-S4", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:10:00Z", period_seconds=600, tolerance_seconds=60,
    ).json()
    assert body["covered_count"] == 2
    assert body["gaps"] == []


def test_readings_outside_extended_range_ignored(client):
    _post(client, [
        _reading("CASE-S5", "2026-08-31T20:00:00Z"),
        _reading("CASE-S5", "2026-09-01T05:00:00Z"),
    ])
    body = _coverage(client, "CASE-S5", **BASE).json()
    assert body["covered_count"] == 0
    assert body["coverage"] == pytest.approx(0.0)
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:00:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 7}
    ]


# ---------- 一对一匹配 ----------


def test_reading_prefers_earliest_coverable_slot(client):
    # 时隙 00:00 / 00:10，容差 400s：两窗口在 00:03:20…00:06:40 重叠。
    # 00:05 的读数必须先覆盖 00:00（最早可覆盖），00:07 的读数再覆盖 00:10；
    # 若 00:05 错配 00:10，00:07 将无法覆盖 00:00（窗口止于 00:06:40）。
    _post(client, [
        _reading("CASE-S6", "2026-09-01T00:05:00Z"),
        _reading("CASE-S6", "2026-09-01T00:07:00Z"),
    ])
    body = _coverage(
        client, "CASE-S6", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:10:00Z", period_seconds=600, tolerance_seconds=400,
    ).json()
    assert body["covered_count"] == 2
    assert body["gaps"] == []


def test_one_reading_covers_at_most_one_slot(client):
    _post(client, [_reading("CASE-S7", "2026-09-01T00:00:00Z")])
    body = _coverage(client, "CASE-S7", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["expected_count"] == 7
    assert body["covered_count"] == 1
    assert body["coverage"] == pytest.approx(1 / 7)
    # 缺测时隙 00:10…01:00 连续合并为一个缺口
    assert body["gap_count"] == 1
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:10:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 6}
    ]


def test_slot_accepts_at_most_one_reading(client):
    _post(client, [
        _reading("CASE-S8", "2026-08-31T23:59:30Z"),
        _reading("CASE-S8", "2026-09-01T00:00:30Z"),
    ])
    body = _coverage(
        client, "CASE-S8", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00Z", period_seconds=600, tolerance_seconds=60,
    ).json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1
    assert body["coverage"] == pytest.approx(1.0)


def test_merged_readings_participate(client):
    # 同时间戳重复上报合并为一条读数；合并读数只能覆盖一个时隙
    _post(client, [_reading("CASE-S9", "2026-09-01T00:00:00Z")])
    _post(client, [_reading("CASE-S9", "2026-09-01T00:00:00Z", 21.0, 51.0)])
    body = _coverage(client, "CASE-S9", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["covered_count"] == 1


# ---------- 缺口合并 ----------


def test_consecutive_missing_slots_merge_into_gaps(client):
    # 7 个时隙；覆盖 00:00 / 00:10 / 00:40 → 缺 00:20/00:30 与 00:50/01:00 两段
    _post(client, [
        _reading("CASE-S10", "2026-09-01T00:00:00Z"),
        _reading("CASE-S10", "2026-09-01T00:10:00Z"),
        _reading("CASE-S10", "2026-09-01T00:40:00Z"),
    ])
    body = _coverage(client, "CASE-S10", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["expected_count"] == 7
    assert body["covered_count"] == 3
    assert body["coverage"] == pytest.approx(3 / 7)
    assert body["gap_count"] == 2
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:20:00Z",
         "end_ts": "2026-09-01T00:30:00Z", "missing_slots": 2},
        {"start_ts": "2026-09-01T00:50:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 2},
    ]


def test_isolated_missing_slots_form_single_slot_gaps(client):
    # 覆盖 00:00 / 00:20 / 00:40 → 缺 00:10、00:30、00:50+01:00
    _post(client, [_reading("CASE-S11", _ts(m)) for m in (0, 20, 40)])
    body = _coverage(client, "CASE-S11", **{**BASE, "tolerance_seconds": 30}).json()
    assert body["covered_count"] == 3
    assert [g["missing_slots"] for g in body["gaps"]] == [1, 1, 2]
    assert [(g["start_ts"], g["end_ts"]) for g in body["gaps"]] == [
        ("2026-09-01T00:10:00Z", "2026-09-01T00:10:00Z"),
        ("2026-09-01T00:30:00Z", "2026-09-01T00:30:00Z"),
        ("2026-09-01T00:50:00Z", "2026-09-01T01:00:00Z"),
    ]


# ---------- 404 / 422 ----------


def test_unknown_case_404(client):
    r = _coverage(client, "NOPE", **BASE)
    assert r.status_code == 404


def test_422_start_after_end(client):
    _post(client, [_reading("CASE-S12", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S12", start="2026-09-01T01:00:00Z",
        end="2026-09-01T00:00:00Z", period_seconds=600, tolerance_seconds=60,
    )
    assert r.status_code == 422


def test_422_invalid_period_and_tolerance(client):
    _post(client, [_reading("CASE-S13", "2026-09-01T00:00:00Z")])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z"}
    # 周期必须为正
    assert _coverage(client, "CASE-S13", **base, period_seconds=0,
                     tolerance_seconds=0).status_code == 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=-600,
                     tolerance_seconds=0).status_code == 422
    # 容差非负
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=-1).status_code == 422
    # 容差必须严格小于周期：等于、大于均 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=600).status_code == 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=601).status_code == 422
    # 缺必填参数
    assert client.get("/api/v1/cases/CASE-S13/sampling-coverage").status_code == 422


# ---------- 周期量化为整数微秒 & 时隙数量边界 ----------


def test_422_sub_microsecond_period(client):
    _post(client, [_reading("CASE-S15", "2026-09-01T00:00:00Z")])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:00:01Z"}
    # 不足 1 微秒的周期（量化后为 0 微秒）一律 422，不再生成重复微秒时隙
    for p in (0.0000004, 0.000000499999, 5e-7):
        r = _coverage(client, "CASE-S15", **base,
                      period_seconds=p, tolerance_seconds=0)
        assert r.status_code == 422, (p, r.status_code, r.text)


def test_period_quantized_to_integer_microseconds(client):
    # 0.0000015 四舍五入为 2µs；窗口 10µs → 时隙 0/2/4/6/8/10 共 6 个，
    # 不再出现浮点累加导致的重复微秒时隙
    _post(client, [
        _reading("CASE-S16", f"2026-09-01T00:00:00.{i:06d}Z")
        for i in range(0, 11, 2)
    ])
    r = _coverage(
        client, "CASE-S16",
        start="2026-09-01T00:00:00.000000Z",
        end="2026-09-01T00:00:00.000010Z",
        period_seconds=0.0000015, tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period_seconds"] == 0.000002
    assert body["expected_count"] == 6
    assert body["covered_count"] == 6
    assert body["gaps"] == []


def test_sub_microsecond_period_no_duplicate_slots_unit():
    # 单元层：0.4µs 周期在旧实现下会产出数百万重复微秒时隙；
    # 新实现先算数，且该周期在接口层即判 422（normalize 抛错）
    from app.sampling import SamplingParamsError, generate_slots, normalize_period

    with pytest.raises(SamplingParamsError):
        normalize_period(0.0000004)

    start = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(seconds=1)
    # 1µs 周期：1 秒窗口恰好 1000001 个 → 超 100000 上限
    with pytest.raises(SamplingParamsError):
        generate_slots(start, end, 1)
    # 2µs 周期、0.1 秒窗口：50001 个时隙且全部唯一（整数微秒，无重复）
    short_end = start + dt.timedelta(microseconds=100_000)
    slots = generate_slots(start, short_end, 2)
    assert len(slots) == 50001
    assert len(set(slots)) == len(slots)


def test_period_larger_than_window_single_slot_no_overflow(client):
    # 极大正周期（远超 timedelta 可表示范围）+ 更短查询窗：
    # 只生成起点时隙，不试算下一个时隙，旧实现会 OverflowError → 500
    _post(client, [_reading("CASE-S17", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S17", start="2026-09-01T00:00:00Z",
        end="2026-09-01T01:00:00Z", period_seconds=1e300, tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1
    assert body["coverage"] == pytest.approx(1.0)
    assert body["gaps"] == []


def test_period_larger_than_window_reading_outside_tolerance(client):
    # 周期大于窗口：起点时隙仍按容差窗口匹配，容差外读数不覆盖
    _post(client, [_reading("CASE-S18", "2026-09-01T00:05:00Z")])
    r = _coverage(
        client, "CASE-S18", start="2026-09-01T00:00:00Z",
        end="2026-09-01T01:00:00Z", period_seconds=100000,
        tolerance_seconds=60,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 0
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:00:00Z",
         "end_ts": "2026-09-01T00:00:00Z", "missing_slots": 1}
    ]


def test_422_expected_slots_over_limit(client):
    _post(client, [_reading("CASE-S19", "2026-09-01T00:00:00Z")])
    # 1µs 周期、1 秒窗口 → 1000001 个时隙，超过 100000 上限 → 422
    r = _coverage(
        client, "CASE-S19", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:01Z", period_seconds=0.000001,
        tolerance_seconds=0,
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["limit"] == 100000
    assert detail["actual"] == 1000001


def test_expected_slots_at_limit_accepted(client):
    # 窗口 99999µs、周期 1µs → 恰好 100000 个时隙，边界值接受
    _post(client, [_reading("CASE-S20", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S20", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00.099999Z", period_seconds=0.000001,
        tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    assert r.json()["expected_count"] == 100000


# ---------- 只读 & 确定性 ----------


def test_read_only_and_deterministic(client, db):
    _post(client, [_reading("CASE-S14", _ts(m)) for m in range(0, 61, 10)])

    state = db.get(CaseState, "CASE-S14")
    version_before = state.version
    readings_before = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading)
        .filter(Reading.device_sn == "CASE-S14")
        .order_by(Reading.ts)
    ]
    snaps_before = db.query(SegmentSnapshot).filter(
        SegmentSnapshot.device_sn == "CASE-S14"
    ).count()

    r1 = _coverage(client, "CASE-S14", **BASE).json()
    r2 = _coverage(client, "CASE-S14", **BASE).json()
    assert r1 == r2

    db.expire_all()
    assert db.get(CaseState, "CASE-S14").version == version_before
    readings_after = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading)
        .filter(Reading.device_sn == "CASE-S14")
        .order_by(Reading.ts)
    ]
    assert readings_after == readings_before
    assert (
        db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == "CASE-S14")
        .count()
        == snaps_before
    )
