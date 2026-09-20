"""保管容量并发审计接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.custody import CustodyCapacityAuditError, audit_custody_capacity
from app.main import app
from app.models import (
    ChainEvents,
    CustodianCapacity,
    CustodyCapacityAuditRequest,
    Event,
    EventType,
)
client = TestClient(app)

DAY = "2026-09-19"


def dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:{second:02d}+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def chain(chain_id: str, events) -> ChainEvents:
    return ChainEvents(chain_id=chain_id, events=list(events))


def cap(custodian: str, capacity: int) -> CustodianCapacity:
    return CustodianCapacity(custodian=custodian, capacity=capacity)


def make_request(chains, capacities, audit_until=None) -> CustodyCapacityAuditRequest:
    return CustodyCapacityAuditRequest(
        chains=list(chains),
        capacities=list(capacities),
        audit_until=audit_until if audit_until is not None else dt(20),
    )


def segments_of(resp):
    return [
        (s.custodian, s.start, s.end, s.capacity, s.concurrency, tuple(s.chain_ids))
        for s in resp.segments
    ]


def summary_map(resp):
    return {c.custodian: c for c in resp.custodians}


ALICE_AND_BOB = [cap("Alice", 1), cap("Bob", 5)]


# ---------------------------------------------------------------------------
# 基本超限识别
# ---------------------------------------------------------------------------


def test_basic_overrun():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1)],
        )
    )
    # chain-a [08,20)、chain-b [09,20)：[08,09) 并发 1 不超限，[09,20) 并发 2 超限
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(20), 1, 2, ("chain-a", "chain-b")),
    ]
    summary = summary_map(resp)["Alice"]
    assert summary.capacity == 1
    assert summary.overrun_seconds == 11 * 3600
    assert summary.peak_concurrency == 2


def test_concurrency_within_capacity_no_segments():
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
    assert summary_map(resp)["Alice"].overrun_seconds == 0.0
    assert summary_map(resp)["Alice"].peak_concurrency == 2


def test_overrun_only_beyond_capacity():
    # 容量 2：三条链 [08,20) / [09,20) / [10,11) 仅 [10,11) 并发 3 超限
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
                chain(
                    "chain-c",
                    [
                        ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice"),
                        ev(EventType.HANDOVER, 11, custodian="Bob"),
                    ],
                ),
            ],
            capacities=[cap("Alice", 2), cap("Bob", 5)],
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(10), dt(11), 2, 3, ("chain-a", "chain-b", "chain-c")),
    ]
    assert summary_map(resp)["Alice"].peak_concurrency == 3
    assert summary_map(resp)["Bob"].peak_concurrency == 1


# ---------------------------------------------------------------------------
# 扫描线边界：同刻先结束后开始（半开区间）
# ---------------------------------------------------------------------------


def test_same_instant_end_before_start_not_concurrent():
    # chain-a 在 Alice 名下 [08,10)，10 点交接给 Bob；chain-b 10 点起由 Alice 持有
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 10, custodian="Bob"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 10, seal_id="S2", custodian="Alice")]),
            ],
            capacities=ALICE_AND_BOB,
        )
    )
    assert resp.segments == []
    assert summary_map(resp)["Alice"].peak_concurrency == 1
    assert summary_map(resp)["Bob"].peak_concurrency == 1


def test_overlapping_boundary_order_within_same_custodian():
    # 同刻先结束后开始：chain-b 在 Alice [09,10)，chain-a [08,20)，
    # chain-c [10,20) 加入：[09,10) 集合 {a,b}、[10,20) 集合 {a,c}
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.HANDOVER, 10, custodian="Bob"),
                    ],
                ),
                chain("chain-c", [ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice")]),
            ],
            capacities=ALICE_AND_BOB,
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(10), 1, 2, ("chain-a", "chain-b")),
        ("Alice", dt(10), dt(20), 1, 2, ("chain-a", "chain-c")),
    ]


# ---------------------------------------------------------------------------
# 超限段合并
# ---------------------------------------------------------------------------


def test_adjacent_same_active_set_merged():
    # chain-a 在 10 点做一次接收人为同一保管人的合法 handover：Alice 区间在 10 点
    # 结束又开始，两侧活跃集合均为 {chain-a, chain-b} → 合并为一段 [09,20)
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 10, custodian="Alice"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1)],
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(20), 1, 2, ("chain-a", "chain-b")),
    ]
    assert resp.segments[0].seconds == 11 * 3600


def test_active_set_change_splits_segments():
    # {a,b} → {a,b,c} → {a,c}：相邻但活跃集合不同，不合并
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.HANDOVER, 11, custodian="Bob"),
                    ],
                ),
                chain("chain-c", [ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice")]),
            ],
            capacities=ALICE_AND_BOB,
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(10), 1, 2, ("chain-a", "chain-b")),
        ("Alice", dt(10), dt(11), 1, 3, ("chain-a", "chain-b", "chain-c")),
        ("Alice", dt(11), dt(20), 1, 2, ("chain-a", "chain-c")),
    ]


def test_non_overrun_gap_breaks_merging():
    # 同集合 {a,b} 的两段超限 [09,10) 与 [12,20) 之间夹着并发 1 的区间：不合并
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.HANDOVER, 10, custodian="Bob"),
                        ev(EventType.HANDOVER, 12, custodian="Alice"),
                    ],
                ),
            ],
            capacities=ALICE_AND_BOB,
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(10), 1, 2, ("chain-a", "chain-b")),
        ("Alice", dt(12), dt(20), 1, 2, ("chain-a", "chain-b")),
    ]
    assert summary_map(resp)["Alice"].overrun_seconds == 9 * 3600


# ---------------------------------------------------------------------------
# 持有区间推导：违规事件 / unseal / 末段延伸 / audit_until 截断
# ---------------------------------------------------------------------------


def test_violation_events_produce_no_intervals():
    # chain-b 的第二次 seal 在已封装状态下违规，不产生区间
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 9, custodian="Alice"),
                    ],
                ),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.SEAL, 10, seal_id="S3", custodian="Alice"),
                    ],
                ),
            ],
            capacities=[cap("Alice", 1)],
        )
    )
    assert resp.chains[1].valid is False
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(20), 1, 2, ("chain-a", "chain-b")),
    ]


def test_unseal_does_not_change_custodian():
    # chain-a：Alice [08,09)、开封、Bob 09 点 unseal 后重封为 Bob；
    # Alice 只持有 [08,09)，chain-b 由 Alice [09,20) 持有，同刻交接不并发
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.UNSEAL, 8, 30),
                        ev(EventType.RESEAL, 9, seal_id="S2", custodian="Bob"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S3", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1), cap("Bob", 1)],
        )
    )
    assert resp.segments == []
    assert set(summary_map(resp)) == {"Alice", "Bob"}


def test_audit_until_clips_last_interval():
    # 末个区间延伸至 audit_until=12 而非 20
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            capacities=[cap("Alice", 1)],
            audit_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(9), dt(12), 1, 2, ("chain-a", "chain-b")),
    ]
    assert summary_map(resp)["Alice"].overrun_seconds == 3 * 3600


def test_zero_length_interval_at_audit_until_dropped():
    # 最后一个有效事件恰在 audit_until：末个区间退化为零长度，不产生任何持有区间
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 10, custodian="Bob"),
                    ],
                )
            ],
            capacities=ALICE_AND_BOB,
            audit_until=dt(10),
        )
    )
    assert resp.segments == []
    # Bob 名下为零长度区间 → 不参与审计，汇总中只有 Alice
    assert set(summary_map(resp)) == {"Alice"}
    assert summary_map(resp)["Alice"].peak_concurrency == 1


def test_empty_chains_and_capacities():
    resp = audit_custody_capacity(make_request(chains=[], capacities=[]))
    assert resp.chains == []
    assert resp.segments == []
    assert resp.custodians == []


# ---------------------------------------------------------------------------
# 多保管人汇总与排序
# ---------------------------------------------------------------------------


def test_segments_and_custodians_ordering():
    # 超限保管人 Bob 字典序在 Alice 之后；分段按 (保管人, 起点) 升序
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 12, custodian="Bob"),
                    ],
                ),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.HANDOVER, 13, custodian="Bob"),
                    ],
                ),
            ],
            capacities=[cap("Alice", 1), cap("Bob", 1)],
        )
    )
    custodian_order = [s.custodian for s in resp.segments]
    assert custodian_order == sorted(custodian_order)
    assert [c.custodian for c in resp.custodians] == ["Alice", "Bob"]
    # Alice 超限 [09,12) 共 3 小时；Bob 超限 [13,20) 共 7 小时
    assert summary_map(resp)["Alice"].overrun_seconds == 3 * 3600
    assert summary_map(resp)["Bob"].overrun_seconds == 7 * 3600


def test_extra_capacity_for_inactive_custodian_ignored():
    resp = audit_custody_capacity(
        make_request(
            chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
            capacities=[cap("Alice", 1), cap("Zoe", 9)],
        )
    )
    assert set(summary_map(resp)) == {"Alice"}


def test_chain_validation_results_preserved_in_input_order():
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain(
                    "chain-b",
                    [ev(EventType.HANDOVER, 9, custodian="Bob")],  # 未封装交接，违规
                ),
            ],
            capacities=[cap("Alice", 1)],
        )
    )
    assert [c.chain_id for c in resp.chains] == ["chain-a", "chain-b"]
    assert resp.chains[0].valid is True
    assert resp.chains[1].valid is False
    # 违规链无任何有效保管事件，不要求 Bob 容量
    assert set(summary_map(resp)) == {"Alice"}


# ---------------------------------------------------------------------------
# 422：缺少容量 / 容量重复 / 混用时区 / audit_until 过早 / chain_id 重复
# ---------------------------------------------------------------------------


def test_missing_capacity_raises_422():
    with pytest.raises(CustodyCapacityAuditError, match="缺少保管人容量"):
        audit_custody_capacity(
            make_request(
                chains=[
                    chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                    chain(
                        "chain-b",
                        [
                            ev(EventType.SEAL, 8, seal_id="S2", custodian="Bob"),
                            ev(EventType.HANDOVER, 9, custodian="Alice"),
                        ],
                    ),
                ],
                capacities=[cap("Alice", 1)],
            )
        )


def test_duplicate_capacity_raises_422():
    with pytest.raises(CustodyCapacityAuditError, match="保管人容量重复"):
        audit_custody_capacity(
            make_request(
                chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
                capacities=[cap("Alice", 1), cap("Alice", 2)],
            )
        )


def test_audit_until_equal_to_last_valid_event_allowed():
    # 恰好相等允许；该末个区间退化为零长度并丢弃
    resp = audit_custody_capacity(
        make_request(
            chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
            capacities=[cap("Alice", 1)],
            audit_until=dt(8),
        )
    )
    assert resp.segments == []
    assert resp.custodians == []


def test_audit_until_before_valid_event_raises_422():
    with pytest.raises(CustodyCapacityAuditError, match="audit_until 早于最后一个有效事件"):
        audit_custody_capacity(
            make_request(
                chains=[chain("chain-a", [ev(EventType.SEAL, 18, seal_id="S1", custodian="Alice")])],
                capacities=[cap("Alice", 1)],
                audit_until=dt(10),
            )
        )


def test_violation_event_does_not_block_earlier_audit_until():
    # 8 点有效 seal，18 点违规 handover（未封装状态下不得交接）：
    # 违规事件不推进有效时间基准，audit_until=10 仍合法
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.UNSEAL, 9),
                        ev(EventType.HANDOVER, 18, custodian="Bob"),  # 未封装交接，违规
                    ],
                )
            ],
            capacities=[cap("Alice", 1)],
            audit_until=dt(10),
        )
    )
    assert resp.chains[0].valid is False
    # Alice 仅持有 [08,09)
    assert summary_map(resp)["Alice"].peak_concurrency == 1
    assert set(summary_map(resp)) == {"Alice"}


def test_violation_after_last_valid_does_not_advance_baseline():
    # 8 点有效 seal，18 点违规 reseal（未开封）：audit_until=10 合法
    resp = audit_custody_capacity(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.UNSEAL, 9),
                        ev(EventType.RESEAL, 18, seal_id="S1", custodian="Alice"),  # 复用违规
                    ],
                )
            ],
            capacities=[cap("Alice", 1)],
            audit_until=dt(10),
        )
    )
    # 开封后未有效重封：Alice 持有 [08,09)
    assert summary_map(resp)["Alice"].peak_concurrency == 1


def test_mixed_timezone_raises_422():
    # 事件带时区而 audit_until 不带时区：审计计算阶段报 422
    request = CustodyCapacityAuditRequest.model_validate(
        {
            "chains": [
                {
                    "chain_id": "chain-a",
                    "events": [
                        {
                            "type": "seal",
                            "timestamp": f"{DAY}T08:00:00Z",
                            "seal_id": "S1",
                            "custodian": "Alice",
                        }
                    ],
                }
            ],
            "capacities": [{"custodian": "Alice", "capacity": 1}],
            "audit_until": f"{DAY}T20:00:00",
        }
    )
    with pytest.raises(CustodyCapacityAuditError, match="时区"):
        audit_custody_capacity(request)


# ---------------------------------------------------------------------------
# 模型层校验（pydantic → 422 结构错误）
# ---------------------------------------------------------------------------


def test_duplicate_chain_id_rejected_by_model():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_request(
            chains=[
                chain("chain-a", []),
                chain("chain-a", []),
            ],
            capacities=[],
        )


def test_non_positive_capacity_rejected_by_model():
    from pydantic import ValidationError

    for bad in (0, -1):
        with pytest.raises(ValidationError):
            CustodianCapacity(custodian="Alice", capacity=bad)


# ---------------------------------------------------------------------------
# HTTP 集成测试
# ---------------------------------------------------------------------------


def _http_payload():
    return {
        "chains": [
            {
                "chain_id": "chain-a",
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                     "seal_id": "S001", "custodian": "Alice"}
                ],
            },
            {
                "chain_id": "chain-b",
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
                     "seal_id": "S002", "custodian": "Alice"}
                ],
            },
        ],
        "capacities": [{"custodian": "Alice", "capacity": 1}],
        "audit_until": f"{DAY}T12:00:00Z",
    }


def test_http_endpoint_success():
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=_http_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["audit_until"] == f"{DAY}T12:00:00Z"
    assert [c["chain_id"] for c in data["chains"]] == ["chain-a", "chain-b"]
    assert data["segments"] == [
        {
            "custodian": "Alice",
            "start": f"{DAY}T09:00:00Z",
            "end": f"{DAY}T12:00:00Z",
            "seconds": 10800.0,
            "capacity": 1,
            "concurrency": 2,
            "chain_ids": ["chain-a", "chain-b"],
        }
    ]
    assert data["custodians"] == [
        {"custodian": "Alice", "capacity": 1,
         "overrun_seconds": 10800.0, "peak_concurrency": 2}
    ]


def test_http_missing_capacity_returns_422():
    payload = _http_payload()
    payload["capacities"] = []
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422
    assert "缺少保管人容量" in response.json()["detail"]


def test_http_duplicate_capacity_returns_422():
    payload = _http_payload()
    payload["capacities"] = [
        {"custodian": "Alice", "capacity": 1},
        {"custodian": "Alice", "capacity": 2},
    ]
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422
    assert "保管人容量重复" in response.json()["detail"]


def test_http_mixed_timezone_returns_422():
    payload = _http_payload()
    payload["audit_until"] = f"{DAY}T12:00:00"  # naive
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422
    assert "时区" in response.json()["detail"]


def test_http_audit_until_before_valid_event_returns_422():
    payload = _http_payload()
    payload["audit_until"] = f"{DAY}T07:00:00Z"
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422
    assert "audit_until" in response.json()["detail"]


def test_http_duplicate_chain_id_returns_422():
    payload = _http_payload()
    payload["chains"][1]["chain_id"] = "chain-a"
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422


def test_http_non_positive_capacity_returns_422():
    payload = _http_payload()
    payload["capacities"] = [{"custodian": "Alice", "capacity": 0}]
    response = client.post("/api/v1/seal-chain/custody-capacity-audit", json=payload)
    assert response.status_code == 422
