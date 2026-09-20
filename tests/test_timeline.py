"""时间线公共层（app.timeline）的参数化回归测试。

覆盖三个审计端点共享的时间线逻辑：违规事件跳过（保留全链下标）、
嵌套 / 相接窗口、交接点切分、未归属段与混合时区；验证迁移后公开行为
（响应排序、浮点秒数、422 detail）保持不变。
"""
from __future__ import annotations

import itertools
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.exposure import ExposureAuditError, audit_exposure
from app.handover import HandoverPlanAuditError, audit_handover_plan
from app.main import app
from app.models import (
    AllowedWindow,
    Event,
    EventType,
    ExposureAuditRequest,
    HandoverPlanAuditRequest,
    HandoverPlanNode,
    TemperatureAuditRequest,
    TemperatureSample,
)
from app.temperature import TemperatureAuditError, audit_temperature
from app.timeline import (
    MIXED_TIMEZONE_MESSAGE,
    CoverInterval,
    ensure_comparable,
    ok_event_records,
    split_by_cover,
)
from app.validator import validate_events

client = TestClient(app)

DAY = "2026-09-19"


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:00+00:00")


def naive_dt(hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def smp(hour: int, minute: int, temperature: float) -> TemperatureSample:
    return TemperatureSample(timestamp=dt(hour, minute), temperature=temperature)


def violating_chain():
    """下标 0/4/7 为违规事件，ok 下标为 {1, 2, 3, 5, 6} 的事件链。"""
    return [
        ev(EventType.UNSEAL, 8),                                        # 违规：未封装开封
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),         # ok
        ev(EventType.HANDOVER, 10, custodian="Bob"),                    # ok
        ev(EventType.UNSEAL, 10, 30),                                   # ok
        ev(EventType.RESEAL, 11, seal_id="S1", custodian="Bob"),        # 违规：复用封签
        ev(EventType.RESEAL, 11, 30, seal_id="S2", custodian="Carol"),  # ok
        ev(EventType.HANDOVER, 12, custodian="Dave"),                   # ok
        ev(EventType.SEAL, 12, 30, seal_id="S3", custodian="Dave"),     # 违规：已封装
    ]


# ---------------------------------------------------------------------------
# ok_event_records：违规事件跳过，保留全链下标
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_types,expected",
    [
        (
            (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER),
            [(1, "seal", "Alice"), (2, "handover", "Bob"),
             (5, "reseal", "Carol"), (6, "handover", "Dave")],
        ),
        (
            (EventType.UNSEAL, EventType.RESEAL),
            [(3, "unseal", None), (5, "reseal", "Carol")],
        ),
        ((EventType.UNSEAL,), [(3, "unseal", None)]),
        ((EventType.SEAL,), [(1, "seal", "Alice")]),
        ((), []),
    ],
    ids=["custodian_types", "open_interval_types", "unseal_only", "seal_only", "no_types"],
)
def test_ok_event_records_skip_violations(event_types, expected):
    events = violating_chain()
    results = validate_events(events).events
    assert [r.index for r in results if r.status != "ok"] == [0, 4, 7]

    records = ok_event_records(events, results, event_types)
    assert [(r.index, r.type.value, r.custodian) for r in records] == expected
    # 记录与源事件一一对应：时间戳取自全链原事件
    for record in records:
        assert record.timestamp == events[record.index].timestamp


# ---------------------------------------------------------------------------
# split_by_cover：covered / gap 切分（相接边界、开放末端、未归属段）
# ---------------------------------------------------------------------------

# 目标区间 [10:00, 12:00)；cover 记为 (起点, 终点或 None, payload)，
# 期望片段记为 (起点, 终点, covered, payload)
_SPLIT_CASES = [
    (
        "no_cover_all_gap",
        [],
        [((10, 0), (12, 0), False, None)],
    ),
    (
        "exact_cover",
        [((10, 0), (12, 0), "A")],
        [((10, 0), (12, 0), True, "A")],
    ),
    (
        "abutting_covers_split_at_shared_bound",
        [((10, 0), (11, 0), "A"), ((11, 0), (12, 0), "B")],
        [((10, 0), (11, 0), True, "A"), ((11, 0), (12, 0), True, "B")],
    ),
    (
        "open_ended_cover_reaches_target_end",
        [((11, 0), None, "A")],
        [((10, 0), (11, 0), False, None), ((11, 0), (12, 0), True, "A")],
    ),
    (
        "gaps_before_between_and_after",
        [((10, 30), (11, 0), "A"), ((11, 30), None, "B")],
        [
            ((10, 0), (10, 30), False, None),
            ((10, 30), (11, 0), True, "A"),
            ((11, 0), (11, 30), False, None),
            ((11, 30), (12, 0), True, "B"),
        ],
    ),
    (
        "cover_entirely_before_target",
        [((8, 0), (9, 0), "A")],
        [((10, 0), (12, 0), False, None)],
    ),
    (
        "cover_entirely_after_target",
        [((13, 0), (14, 0), "A")],
        [((10, 0), (12, 0), False, None)],
    ),
    (
        "covers_clipped_to_target_bounds",
        [((9, 0), (10, 30), "A"), ((11, 0), (13, 0), "B")],
        [
            ((10, 0), (10, 30), True, "A"),
            ((10, 30), (11, 0), False, None),
            ((11, 0), (12, 0), True, "B"),
        ],
    ),
]


@pytest.mark.parametrize(
    "covers,expected",
    [pytest.param(covers, expected, id=name) for name, covers, expected in _SPLIT_CASES],
)
def test_split_by_cover(covers, expected):
    cover_intervals = [
        CoverInterval(
            start=dt(*start), end=None if end is None else dt(*end), payload=payload
        )
        for start, end, payload in covers
    ]
    pieces = split_by_cover(dt(10, 0), dt(12, 0), cover_intervals)

    assert [
        ((p.start.hour, p.start.minute), (p.end.hour, p.end.minute),
         p.covered, p.payload)
        for p in pieces
    ] == expected
    # 片段按时间升序且无缝拼回目标区间；浮点秒数等于 end - start
    cursor = dt(10, 0)
    for piece in pieces:
        assert piece.start == cursor
        assert piece.seconds == (piece.end - piece.start).total_seconds()
        cursor = piece.end
    assert cursor == dt(12, 0)


def test_split_by_cover_degenerate_target_produces_nothing():
    covers = [CoverInterval(start=dt(9), end=dt(11), payload="A")]
    assert split_by_cover(dt(10), dt(10), covers) == []


# ---------------------------------------------------------------------------
# ensure_comparable：统一时区可比性检查（混合时区）
# ---------------------------------------------------------------------------

_AUDIT_ERRORS = [
    pytest.param(ExposureAuditError, id="exposure"),
    pytest.param(HandoverPlanAuditError, id="handover"),
    pytest.param(TemperatureAuditError, id="temperature"),
]


@pytest.mark.parametrize("error_type", _AUDIT_ERRORS)
@pytest.mark.parametrize(
    "values",
    [
        [dt(10), naive_dt(11)],
        [naive_dt(10), dt(11)],
        [dt(10), dt(11), naive_dt(12)],
    ],
    ids=["aware_then_naive", "naive_then_aware", "trailing_naive"],
)
def test_ensure_comparable_mixed_timezone_raises(error_type, values):
    with pytest.raises(error_type) as exc_info:
        ensure_comparable(values, error_type)
    assert str(exc_info.value) == MIXED_TIMEZONE_MESSAGE


@pytest.mark.parametrize(
    "values",
    [
        [],
        [dt(10)],
        [naive_dt(10)],
        [dt(10), dt(11)],
        [naive_dt(10), naive_dt(11)],
    ],
    ids=["empty", "single_aware", "single_naive", "all_aware", "all_naive"],
)
def test_ensure_comparable_consistent_timezone_passes(values):
    ensure_comparable(values, ValueError)  # 不抛异常


# ---------------------------------------------------------------------------
# 端点回归：违规事件跳过（三个端点，全链下标）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint,expected_indices",
    [
        ("exposure", {3, 5}),
        ("handover", {1, 2, 5, 6}),
        ("temperature", {1, 2, 5}),
    ],
)
def test_endpoints_skip_violations_and_keep_full_chain_indices(
    endpoint, expected_indices
):
    events = violating_chain()
    ok_indices = {
        r.index for r in validate_events(events).events if r.status == "ok"
    }
    assert ok_indices == {1, 2, 3, 5, 6}

    if endpoint == "exposure":
        resp = audit_exposure(
            ExposureAuditRequest(events=events, audit_until=dt(13), allowed_windows=[])
        )
        # 唯一开封区间 [10:30, 11:30)：重封下标是合法 reseal 的 5，不是违规的 4
        referenced = {
            index
            for segment in resp.covered + resp.uncovered
            for index in (segment.unseal_index, segment.reseal_index)
            if index is not None
        }
    elif endpoint == "handover":
        plan = [
            HandoverPlanNode(custodian="Alice",
                             window_start=dt(8, 30), window_end=dt(9, 30)),
            HandoverPlanNode(custodian="Bob",
                             window_start=dt(9, 30), window_end=dt(10, 30)),
            HandoverPlanNode(custodian="Carol",
                             window_start=dt(11, 0), window_end=dt(12, 0)),
            HandoverPlanNode(custodian="Dave",
                             window_start=dt(12, 0), window_end=dt(13, 0)),
        ]
        resp = audit_handover_plan(
            HandoverPlanAuditRequest(events=events, plan=plan)
        )
        referenced = {
            r.event_index for r in resp.plan_results if r.event_index is not None
        }
        referenced |= {u.event_index for u in resp.unexpected}
    else:
        samples = [
            smp(9, 30, 10.0), smp(9, 40, 5.0),
            smp(10, 45, 10.0), smp(10, 50, 5.0),
            smp(11, 45, 10.0), smp(11, 50, 5.0),
        ]
        resp = audit_temperature(
            TemperatureAuditRequest(
                events=events,
                samples=samples,
                min_temperature=2.0,
                max_temperature=8.0,
                max_gap_seconds=600.0,
            )
        )
        referenced = {
            s.custodian_event_index
            for s in resp.segments
            if s.custodian_event_index is not None
        }

    assert referenced == expected_indices
    assert referenced <= ok_indices


# ---------------------------------------------------------------------------
# 端点回归：嵌套 / 相接窗口（开封暴露审计）
# ---------------------------------------------------------------------------


def _exposure_chain():
    """seal@09, unseal@10, reseal@20：开封区间 [10:00, 20:00)。"""
    return [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.UNSEAL, 10),
        ev(EventType.RESEAL, 20, seal_id="S2", custodian="Alice"),
    ]


def _window(sh: int, sm: int, eh: int, em: int) -> AllowedWindow:
    return AllowedWindow(start=dt(sh, sm), end=dt(eh, em))


# 嵌套窗口 [11:00,12:00) 被 [10:30,12:30) 包含；[12:30,14:00) 与之相接
_NESTED_ABUTTING = [
    _window(10, 30, 12, 30),
    _window(11, 0, 12, 0),
    _window(12, 30, 14, 0),
]


@pytest.mark.parametrize(
    "windows",
    [list(p) for p in itertools.permutations(_NESTED_ABUTTING)],
    ids=["-".join(f"{w.start.hour}:{w.start.minute:02d}" for w in p)
         for p in itertools.permutations(_NESTED_ABUTTING)],
)
def test_exposure_nested_and_abutting_windows_order_independent(windows):
    resp = audit_exposure(
        ExposureAuditRequest(
            events=_exposure_chain(), audit_until=dt(21), allowed_windows=windows
        )
    )
    # 合并为 [10:30, 14:00)：covered / uncovered 与窗口输入顺序无关
    assert [(s.start, s.end, s.seconds) for s in resp.covered] == [
        (dt(10, 30), dt(14, 0), 12600.0)
    ]
    assert [(s.start, s.end, s.seconds) for s in resp.uncovered] == [
        (dt(10, 0), dt(10, 30), 1800.0),
        (dt(14, 0), dt(20, 0), 21600.0),
    ]
    assert resp.unauthorized_seconds == 23400.0


@pytest.mark.parametrize(
    "windows",
    [
        [_window(11, 0, 12, 0), _window(12, 0, 13, 0)],  # 顺序相接
        [_window(12, 0, 13, 0), _window(11, 0, 12, 0)],  # 乱序相接
    ],
    ids=["ordered", "shuffled"],
)
def test_exposure_abutting_windows_merge_into_single_cover(windows):
    resp = audit_exposure(
        ExposureAuditRequest(
            events=_exposure_chain(), audit_until=dt(21), allowed_windows=windows
        )
    )
    assert [(s.start, s.end) for s in resp.covered] == [(dt(11), dt(13))]


# ---------------------------------------------------------------------------
# 端点回归：交接点切分（温控责任审计）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("second_custodian", ["Bob", "Alice"])
def test_temperature_breach_split_at_handover_point(second_custodian):
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 9, 5, custodian=second_custodian),
    ]
    resp = audit_temperature(
        TemperatureAuditRequest(
            events=events,
            samples=[smp(9, 0, 10.5), smp(9, 10, 5.0)],
            min_temperature=2.0,
            max_temperature=8.0,
            max_gap_seconds=600.0,
        )
    )
    # 越界区间 [09:00,09:10) 跨交接点 09:05，必须拆分（同名交接也拆分）
    assert [
        (s.start, s.end, s.seconds, s.custodian, s.custodian_event_index)
        for s in resp.segments
    ] == [
        (dt(9, 0), dt(9, 5), 300.0, "Alice", 0),
        (dt(9, 5), dt(9, 10), 300.0, second_custodian, 1),
    ]
    assert resp.unattributed_seconds == 0.0


def test_temperature_open_ended_last_interval_attributes_tail():
    events = [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]
    resp = audit_temperature(
        TemperatureAuditRequest(
            events=events,
            samples=[smp(20, 0, 10.0), smp(20, 10, 5.0)],
            min_temperature=2.0,
            max_temperature=8.0,
            max_gap_seconds=600.0,
        )
    )
    # 最后一个保管人区间向右开放：远晚于事件的样本仍归 Alice
    assert [(s.start, s.end, s.custodian) for s in resp.segments] == [
        (dt(20, 0), dt(20, 10), "Alice")
    ]
    assert resp.unattributed_seconds == 0.0


# ---------------------------------------------------------------------------
# 端点回归：未归属段（温控责任审计）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "events,expected_segments,expected_unattributed",
    [
        (
            [],
            [(dt(9, 0), dt(9, 10), 600.0, None, None)],
            600.0,
        ),
        (
            [ev(EventType.SEAL, 9, 30, seal_id="S1", custodian="Alice")],
            [
                (dt(9, 25), dt(9, 30), 300.0, None, None),
                (dt(9, 30), dt(9, 35), 300.0, "Alice", 0),
            ],
            300.0,
        ),
    ],
    ids=["no_custodian_events", "breach_before_first_custodian_event"],
)
def test_temperature_unattributed_segments(
    events, expected_segments, expected_unattributed
):
    samples = (
        [smp(9, 0, 10.0), smp(9, 10, 5.0)]
        if not events
        else [smp(9, 25, 10.0), smp(9, 35, 5.0)]
    )
    resp = audit_temperature(
        TemperatureAuditRequest(
            events=events,
            samples=samples,
            min_temperature=2.0,
            max_temperature=8.0,
            max_gap_seconds=600.0,
        )
    )
    assert [
        (s.start, s.end, s.seconds, s.custodian, s.custodian_event_index)
        for s in resp.segments
    ] == expected_segments
    assert resp.unattributed_seconds == expected_unattributed


# ---------------------------------------------------------------------------
# 端点回归：混合时区 422 detail 不变（三个端点）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,payload",
    [
        (
            "/api/v1/seal-chain/exposure-audit",
            {
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
                     "seal_id": "S1", "custodian": "Alice"}
                ],
                "audit_until": f"{DAY}T20:00:00",  # naive，与事件混用
                "allowed_windows": [],
            },
        ),
        (
            "/api/v1/seal-chain/handover-plan-audit",
            {
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
                     "seal_id": "S1", "custodian": "Alice"}
                ],
                "plan": [
                    {"custodian": "Alice",
                     "window_start": f"{DAY}T09:00:00",  # naive，与事件混用
                     "window_end": f"{DAY}T11:00:00"}
                ],
            },
        ),
        (
            "/api/v1/seal-chain/temperature-audit",
            {
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                     "seal_id": "S1", "custodian": "Alice"}
                ],
                "samples": [
                    {"timestamp": f"{DAY}T09:00:00", "temperature": 10.0},  # naive
                    {"timestamp": f"{DAY}T09:10:00", "temperature": 5.0},
                ],
                "min_temperature": 2.0,
                "max_temperature": 8.0,
                "max_gap_seconds": 600,
            },
        ),
    ],
    ids=["exposure", "handover", "temperature"],
)
def test_api_mixed_timezone_422_detail_unchanged(url, payload):
    r = client.post(url, json=payload)
    assert r.status_code == 422
    assert r.json()["detail"] == MIXED_TIMEZONE_MESSAGE
