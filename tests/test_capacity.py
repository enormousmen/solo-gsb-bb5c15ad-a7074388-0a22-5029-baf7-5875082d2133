"""保管容量审计接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.capacity import (
    CustodyCapacityAuditError,
    _RawOverrun,
    _merge_raw_overruns,
    audit_custody_capacity,
)
from app.main import app
from app.models import (
    ChainEvents,
    CustodianCapacity,
    CustodyCapacityAuditRequest,
    Event,
    EventType,
)

client = TestClient(app)

DAY = "2026-09-20"


def dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:{second:02d}+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def chain(chain_id: str, events) -> ChainEvents:
    return ChainEvents(chain_id=chain_id, events=list(events))


def cap(custodian: str, capacity: int) -> CustodianCapacity:
    return CustodianCapacity(custodian=custodian, capacity=capacity)


def make_request(chains, capacities, audit_until=dt(20)):
    return CustodyCapacityAuditRequest(
        chains=list(chains),
        capacities=list(capacities),
        audit_until=audit_until,
    )


def seg_by_key(resp):
    return {(s.custodian, s.start, s.end): s for s in resp.segments}


def summary_by_id(resp):
    return {item.custodian: item for item in resp.custodians}


# ---------------------------------------------------------------------------
# 基本超限段
# ---------------------------------------------------------------------------


def test_basic_overrun_segment():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ev(EventType.HANDOVER, 12, custodian="Bob"),
                ]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
                chain("chain-c", [ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 2), cap("Bob", 1)],
        )
    )
    # Alice：08-09 并发 1、09-10 并发 2（均不超），10-12 并发 3（超限）
    assert len(resp.segments) == 1
    segment = resp.segments[0]
    assert segment.custodian == "Alice"
    assert segment.start == dt(10)
    assert segment.end == dt(12)
    assert segment.seconds == 7200.0
    assert segment.capacity == 2
    assert segment.concurrency == 3
    assert segment.chain_ids == ["chain-a", "chain-b", "chain-c"]
    # Bob 始终只有 chain-a，不超限
    assert summary_by_id(resp).keys() == {"Alice"}
    assert summary_by_id(resp)["Alice"].overrun_seconds == 7200.0
    assert summary_by_id(resp)["Alice"].peak_concurrency == 3


def test_concurrency_equal_to_capacity_is_not_overrun():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 2)],
        )
    )
    assert resp.segments == []
    assert resp.custodians == []


def test_no_chains_and_no_capacities_returns_empty():
    resp = audit_custody_capacity(make_request(chains=[], capacities=[]))
    assert resp.segments == []
    assert resp.custodians == []
    assert resp.chains == []


def test_capacity_for_unseen_custodian_is_allowed():
    # 为从未持有任何链的保管人配置容量不报错
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1), cap("Zoe", 2)],
        )
    )
    assert resp.segments == []
    assert resp.custodians == []


def test_audit_until_clips_last_holding_interval():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1)],
            audit_until=dt(10),
        )
    )
    # 末个区间延伸至 audit_until：09:00-10:00 并发 2，超 3600 秒
    assert len(resp.segments) == 1
    assert resp.segments[0].start == dt(9)
    assert resp.segments[0].end == dt(10)
    assert resp.segments[0].seconds == 3600.0


# ---------------------------------------------------------------------------
# 边界扫描：同刻先结束后开始（半开区间）
# ---------------------------------------------------------------------------


def test_same_timestamp_end_before_start_is_not_concurrent():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ev(EventType.HANDOVER, 10, custodian="Bob"),
                ]),
                chain("chain-b", [ev(EventType.SEAL, 10, seal_id="S2", custodian="Alice")]),
            ],
            # Alice 名下：chain-a [08:00,10:00) 与 chain-b [10:00,20:00)；10:00 交接
            capacities=[cap("Alice", 1), cap("Bob", 1)],
        )
    )
    # 半开区间：恰好在 10:00 结束与开始，同刻先结束后开始 → 任何时刻并发都不超过 1
    assert resp.segments == []
    assert resp.custodians == []


def test_overrun_at_final_instant_has_zero_duration():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ev(EventType.HANDOVER, 10, custodian="Bob"),
                ]),
                chain("chain-b", [ev(EventType.SEAL, 10, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1), cap("Bob", 1)],
        )
    )
    # 10:00 先结束 chain-a 再开始 chain-b，超限时长为 0：无任何分段
    assert resp.segments == []


def test_overrun_segments_split_when_active_set_changes():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
                chain("chain-c", [
                    ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice"),
                    ev(EventType.HANDOVER, 11, custodian="Bob"),
                ]),
            ],
            # 09-10 并发 2 {a,b}；10-11 并发 3 {a,b,c}；11-20 并发 2 {a,b}
            capacities=[cap("Alice", 1), cap("Bob", 1)],
        )
    )
    assert [(s.start, s.end, s.concurrency) for s in resp.segments] == [
        (dt(9), dt(10), 2),
        (dt(10), dt(11), 3),
        (dt(11), dt(20), 2),
    ]
    assert resp.segments[0].chain_ids == ["chain-a", "chain-b"]
    assert resp.segments[1].chain_ids == ["chain-a", "chain-b", "chain-c"]
    assert resp.segments[2].chain_ids == ["chain-a", "chain-b"]
    summary = summary_by_id(resp)["Alice"]
    assert summary.overrun_seconds == 3600.0 + 3600.0 + 9 * 3600.0
    assert summary.peak_concurrency == 3


def test_segments_sorted_by_custodian_then_time():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Zoe"),
                ]),
                chain("chain-b", [ev(EventType.SEAL, 8, seal_id="S2", custodian="Zoe")]),
                chain("chain-c", [
                    ev(EventType.SEAL, 9, seal_id="S3", custodian="Alice"),
                ]),
                chain("chain-d", [ev(EventType.SEAL, 9, seal_id="S4", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1), cap("Zoe", 1)],
        )
    )
    # 先按保管人字典序（Alice 在 Zoe 前），保管人内按时间升序
    assert [s.custodian for s in resp.segments] == ["Alice", "Zoe"]
    assert [c.custodian for c in resp.custodians] == ["Alice", "Zoe"]


# ---------------------------------------------------------------------------
# 相邻同集合超限段合并
# ---------------------------------------------------------------------------


def test_merge_adjacent_segments_with_same_active_set():
    a = {"x"}
    raw = [
        _RawOverrun(start=dt(8), end=dt(9), active=set(a)),
        _RawOverrun(start=dt(9), end=dt(10), active={"x"}),
        _RawOverrun(start=dt(10), end=dt(11), active={"x", "y"}),
        _RawOverrun(start=dt(11), end=dt(12), active={"x"}),
    ]
    merged = _merge_raw_overruns(raw)
    # 前两段首尾相接且集合相同 → 合并；第三段集合不同，断开；第四段不与前一段相接合并
    assert len(merged) == 3
    assert (merged[0].start, merged[0].end) == (dt(8), dt(10))
    assert merged[0].active == {"x"}
    assert (merged[1].start, merged[1].end) == (dt(10), dt(11))
    assert (merged[2].start, merged[2].end) == (dt(11), dt(12))


def test_gap_between_overrun_segments_is_not_merged():
    raw = [
        _RawOverrun(start=dt(8), end=dt(9), active={"x"}),
        _RawOverrun(start=dt(10), end=dt(11), active={"x"}),
    ]
    merged = _merge_raw_overruns(raw)
    # 首尾不相接（中间 09-10 不超限）→ 不合并
    assert len(merged) == 2


def test_unseal_does_not_change_custodian():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ev(EventType.UNSEAL, 9),
                    ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
                ]),
                chain("chain-b", [ev(EventType.SEAL, 9, 30, seal_id="S3", custodian="Alice")]),
            ],
            # unseal 不改变保管人：09:30-10:00 开封期间 chain-a 仍在 Alice 名下，并发 2
            capacities=[cap("Alice", 1)],
        )
    )
    assert len(resp.segments) == 1
    assert resp.segments[0].start == dt(9, 30)
    assert resp.segments[0].end == dt(20)
    assert resp.segments[0].concurrency == 2
    assert resp.segments[0].chain_ids == ["chain-a", "chain-b"]


def test_violation_events_generate_no_holding_interval():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [
                    ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ev(EventType.UNSEAL, 9),
                    ev(EventType.HANDOVER, 9, 30, custodian="Bob"),  # 违规：未封装
                    ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
                ]),
            ],
            # 违规 handover 不产生 Bob 区间：Bob 无需容量，只配 Alice 即可
            capacities=[cap("Alice", 1)],
        )
    )
    assert resp.segments == []
    chain_a = next(c for c in resp.chains if c.chain_id == "chain-a")
    assert chain_a.valid is False
    assert [v.index for v in chain_a.violations] == [2]


# ---------------------------------------------------------------------------
# 422 语义错误
# ---------------------------------------------------------------------------


def test_missing_capacity_raises_422():
    with pytest.raises(CustodyCapacityAuditError, match="缺少保管人容量配置"):
        audit_custody_capacity(
            make_request(
                chains=[
                    chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                    chain("chain-b", [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Bob"),
                        ev(EventType.HANDOVER, 10, custodian="Alice"),
                    ]),
                ],
                # 只配了 Alice，Bob 缺少容量
                capacities=[cap("Alice", 1)],
            )
        )


def test_duplicate_capacity_custodian_rejected():
    with pytest.raises(ValidationError):
        make_request(
            chains=[],
            capacities=[cap("Alice", 1), cap("Alice", 2)],
        )


def test_non_positive_capacity_rejected():
    with pytest.raises(ValidationError):
        CustodianCapacity(custodian="Alice", capacity=0)


def test_duplicate_chain_id_rejected():
    with pytest.raises(ValidationError):
        make_request(
            chains=[chain("chain-a", []), chain("chain-a", [])],
            capacities=[],
        )


def test_audit_until_before_last_valid_event_raises_422():
    with pytest.raises(CustodyCapacityAuditError, match="早于最后一个有效事件时间"):
        audit_custody_capacity(
            make_request(
                chains=[
                    chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                ],
                capacities=[cap("Alice", 1)],
                audit_until=dt(7),
            )
        )


def test_audit_until_equal_to_last_valid_event_is_allowed():
    # 末个区间退化为零长度并丢弃；无人持有，也无需任何容量
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
            ],
            capacities=[],
            audit_until=dt(8),
        )
    )
    assert resp.segments == []


def test_mixed_timezone_raises_422():
    naive_event = Event(
        type=EventType.SEAL,
        timestamp=datetime.fromisoformat(f"{DAY}T09:00:00"),  # 不带时区
        seal_id="S2",
        custodian="Alice",
    )
    with pytest.raises(CustodyCapacityAuditError, match="时区"):
        audit_custody_capacity(
            make_request(
                chains=[
                    chain("chain-a", [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        naive_event,
                    ]),
                ],
                capacities=[cap("Alice", 1)],
            )
        )


def test_mixed_timezone_audit_until_raises_422():
    naive_until = datetime.fromisoformat(f"{DAY}T20:00:00")  # 不带时区
    with pytest.raises(CustodyCapacityAuditError, match="时区"):
        audit_custody_capacity(
            make_request(
                chains=[
                    chain("chain-a", [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                    ]),
                ],
                capacities=[cap("Alice", 1)],
                audit_until=naive_until,
            )
        )


# ---------------------------------------------------------------------------
# API 集成
# ---------------------------------------------------------------------------


def test_api_custody_capacity_audit_full_response():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
                {"type": "handover", "timestamp": f"{DAY}T12:00:00Z",
                 "custodian": "Bob"},
            ]},
            {"chain_id": "chain-b", "events": [
                {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
                 "seal_id": "S002", "custodian": "Alice"},
            ]},
            {"chain_id": "chain-c", "events": [
                {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
                 "seal_id": "S003", "custodian": "Alice"},
            ]},
        ],
        "capacities": [
            {"custodian": "Alice", "capacity": 2},
            {"custodian": "Bob", "capacity": 1},
        ],
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 200
    body = r.json()

    assert body["audit_until"] == f"{DAY}T20:00:00Z"
    # 各链原校验结果保留（结构同单链接口并附 chain_id），按输入顺序
    assert [c["chain_id"] for c in body["chains"]] == [
        "chain-a", "chain-b", "chain-c",
    ]
    assert all(c["valid"] is True for c in body["chains"])
    assert body["chains"][0]["events"] == [
        {"index": 0, "type": "seal", "status": "ok", "reason": None,
         "cross_chain_conflict": False},
        {"index": 1, "type": "handover", "status": "ok", "reason": None,
         "cross_chain_conflict": False},
    ]

    assert body["segments"] == [
        {
            "custodian": "Alice",
            "start": f"{DAY}T10:00:00Z",
            "end": f"{DAY}T12:00:00Z",
            "seconds": 7200.0,
            "capacity": 2,
            "concurrency": 3,
            "chain_ids": ["chain-a", "chain-b", "chain-c"],
        }
    ]
    assert body["custodians"] == [
        {
            "custodian": "Alice",
            "capacity": 2,
            "overrun_seconds": 7200.0,
            "peak_concurrency": 3,
        }
    ]


def test_api_missing_capacity_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
        ],
        "capacities": [],
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422
    assert "缺少保管人容量配置" in r.json()["detail"]


def test_api_duplicate_capacity_custodian_returns_422():
    payload = {
        "chains": [],
        "capacities": [
            {"custodian": "Alice", "capacity": 1},
            {"custodian": "Alice", "capacity": 2},
        ],
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422


def test_api_non_positive_capacity_returns_422():
    payload = {
        "chains": [],
        "capacities": [{"custodian": "Alice", "capacity": 0}],
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422


def test_api_audit_until_too_early_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
        ],
        "capacities": [{"custodian": "Alice", "capacity": 1}],
        "audit_until": f"{DAY}T07:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422
    assert "早于" in r.json()["detail"]


def test_api_mixed_timezone_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
        ],
        "capacities": [{"custodian": "Alice", "capacity": 1}],
        "audit_until": f"{DAY}T20:00:00",  # 不带时区
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422
    assert "时区" in r.json()["detail"]


def test_api_duplicate_chain_id_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": []},
            {"chain_id": "chain-a", "events": []},
        ],
        "capacities": [],
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert r.status_code == 422


def test_api_missing_required_fields_returns_422():
    r = client.post("/api/v1/seal-chain/custody-capacity-audit", json={"chains": []})
    assert r.status_code == 422
