"""污染传播溯源接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.contamination import ContaminationTraceError, trace_contamination
from app.main import app
from app.models import (
    ChainEvents,
    ContaminationSource,
    ContaminationTraceRequest,
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


def make_request(chains, source_id, contaminated_at, audit_until):
    return ContaminationTraceRequest(
        chains=list(chains),
        source=ContaminationSource(chain_id=source_id, contaminated_at=contaminated_at),
        audit_until=audit_until,
    )


def contaminated_by_id(resp):
    return {item.chain_id: item for item in resp.contaminated}


# ---------------------------------------------------------------------------
# 基本传播：接触窗口与最早污染时刻
# ---------------------------------------------------------------------------


def test_basic_contact_propagates_at_contaminated_at():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(10),
            audit_until=dt(20),
        )
    )
    # 接触窗口 [09:00,20:00)；源 10:00 已污染且接触仍在持续 → chain-b 在 10:00 被污染
    assert contaminated_by_id(resp)["chain-a"].earliest_contaminated_at == dt(10)
    assert contaminated_by_id(resp)["chain-a"].direct_source is None
    assert contaminated_by_id(resp)["chain-a"].path == ["chain-a"]
    target = contaminated_by_id(resp)["chain-b"]
    assert target.earliest_contaminated_at == dt(10)
    assert target.direct_source == "chain-a"
    assert target.path == ["chain-a", "chain-b"]
    assert resp.unreached == []
    # 各链原校验结果保留
    assert [c.valid for c in resp.chains] == [True, True]


def test_propagation_waits_for_contact_start():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(8, 30),
            audit_until=dt(20),
        )
    )
    # 源 08:30 已污染，但接触 09:00 才开始 → chain-b 在接触起点 09:00 被污染
    assert contaminated_by_id(resp)["chain-b"].earliest_contaminated_at == dt(9)


def test_contact_ended_before_contamination_no_propagation():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain(
                    "chain-b",
                    [
                        ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                        ev(EventType.HANDOVER, 11, custodian="Dave"),
                    ],
                ),
            ],
            source_id="chain-a",
            contaminated_at=dt(12),
            audit_until=dt(20),
        )
    )
    # 接触窗口 [09:00,11:00) 在源被污染（12:00）前已结束 → 不传播
    assert [item.chain_id for item in resp.contaminated] == ["chain-a"]
    assert resp.unreached == ["chain-b"]


def test_contact_half_open_boundary():
    chains = [
        chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
        chain(
            "chain-b",
            [
                ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
                ev(EventType.HANDOVER, 11, custodian="Dave"),
            ],
        ),
    ]
    # 源恰好在接触终点 11:00 被污染：半开区间下接触已不再持续 → 不传播
    resp = trace_contamination(
        make_request(chains, "chain-a", dt(11), dt(20))
    )
    assert resp.unreached == ["chain-b"]
    # 早一分钟：接触仍在持续 → 在该时刻传播
    resp = trace_contamination(
        make_request(chains, "chain-a", dt(10, 59), dt(20))
    )
    assert contaminated_by_id(resp)["chain-b"].earliest_contaminated_at == dt(10, 59)


def test_audit_until_clips_last_holding_interval():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(11, 30),
            audit_until=dt(12),
        )
    )
    # 末个持有区间延伸至 audit_until：接触窗口 [09:00,12:00)，11:30 仍在持续
    assert contaminated_by_id(resp)["chain-b"].earliest_contaminated_at == dt(11, 30)


def test_multiple_contacts_earliest_valid_wins():
    resp = trace_contamination(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.HANDOVER, 9, custodian="Bob"),
                        ev(EventType.HANDOVER, 12, custodian="Alice"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 8, 30, seal_id="S2", custodian="Alice")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(8, 45),
            audit_until=dt(20),
        )
    )
    # 两段接触 [08:30,09:00) 与 [12:00,20:00)：源 08:45 已污染 → 经第一段在 08:45 传播
    assert contaminated_by_id(resp)["chain-b"].earliest_contaminated_at == dt(8, 45)


# ---------------------------------------------------------------------------
# 级联与传播路径
# ---------------------------------------------------------------------------


def test_cascade_path_across_chains():
    resp = trace_contamination(
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
                chain("chain-c", [ev(EventType.SEAL, 10, 30, seal_id="S3", custodian="Bob")]),
                chain("chain-d", [ev(EventType.SEAL, 11, seal_id="S4", custodian="Carol")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(9, 30),
            audit_until=dt(20),
        )
    )
    # a→b：接触 [09:00,10:00)，源 09:30 已污染 → b 在 09:30 被污染
    # b→c：Bob 名下接触 [10:30,20:00) → c 在 10:30 被污染；chain-d 无接触未到达
    assert [(item.chain_id, item.earliest_contaminated_at) for item in resp.contaminated] == [
        ("chain-a", dt(9, 30)),
        ("chain-b", dt(9, 30)),
        ("chain-c", dt(10, 30)),
    ]
    target_b = contaminated_by_id(resp)["chain-b"]
    assert target_b.direct_source == "chain-a"
    assert target_b.path == ["chain-a", "chain-b"]
    target_c = contaminated_by_id(resp)["chain-c"]
    assert target_c.direct_source == "chain-b"
    assert target_c.path == ["chain-a", "chain-b", "chain-c"]
    assert resp.unreached == ["chain-d"]


def test_same_earliest_moment_tie_broken_by_chain_id():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-s", [ev(EventType.SEAL, 1, seal_id="S1", custodian="Hub")]),
                chain("chain-x", [ev(EventType.SEAL, 8, seal_id="S4", custodian="Hub")]),
                chain("chain-b", [ev(EventType.SEAL, 6, seal_id="S3", custodian="Hub")]),
                chain("chain-a", [ev(EventType.SEAL, 5, seal_id="S2", custodian="Hub")]),
            ],
            source_id="chain-s",
            contaminated_at=dt(9),
            audit_until=dt(20),
        )
    )
    # 09:00 时全部接触都在持续：四条链最早污染时刻同为 09:00
    by_id = contaminated_by_id(resp)
    assert set(by_id) == {"chain-s", "chain-a", "chain-b", "chain-x"}
    assert all(item.earliest_contaminated_at == dt(9) for item in by_id.values())
    # 同一最早时刻并列时按 chain_id 字典序固定直接来源
    assert by_id["chain-a"].direct_source == "chain-s"
    assert by_id["chain-a"].path == ["chain-s", "chain-a"]
    assert by_id["chain-b"].direct_source == "chain-a"
    assert by_id["chain-b"].path == ["chain-s", "chain-a", "chain-b"]
    assert by_id["chain-x"].direct_source == "chain-a"
    assert by_id["chain-x"].path == ["chain-s", "chain-a", "chain-x"]
    # contaminated 按 (最早污染时刻, chain_id) 升序
    assert [item.chain_id for item in resp.contaminated] == [
        "chain-a",
        "chain-b",
        "chain-s",
        "chain-x",
    ]


def test_unseal_does_not_change_custodian():
    resp = trace_contamination(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.UNSEAL, 9),
                        ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 9, 30, seal_id="S3", custodian="Alice")]),
            ],
            source_id="chain-b",
            contaminated_at=dt(9, 45),
            audit_until=dt(20),
        )
    )
    # unseal 不改变保管人：开封期间 chain-a 仍在 Alice 名下，接触 [09:30,10:00) 成立
    target = contaminated_by_id(resp)["chain-a"]
    assert target.earliest_contaminated_at == dt(9, 45)
    assert target.direct_source == "chain-b"
    assert target.path == ["chain-b", "chain-a"]


# ---------------------------------------------------------------------------
# 仅 status=ok 的事件生成持有区间
# ---------------------------------------------------------------------------


def test_violation_events_generate_no_holding_interval():
    resp = trace_contamination(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                        ev(EventType.UNSEAL, 9),
                        ev(EventType.HANDOVER, 9, 30, custodian="Bob"),  # 违规：未封装
                        ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
                    ],
                ),
                chain("chain-b", [ev(EventType.SEAL, 9, 15, seal_id="S9", custodian="Bob")]),
            ],
            source_id="chain-b",
            contaminated_at=dt(9, 30),
            audit_until=dt(20),
        )
    )
    # 违规 handover 不产生 Bob 区间：chain-a 始终归 Alice，两链无接触
    assert [item.chain_id for item in resp.contaminated] == ["chain-b"]
    assert resp.unreached == ["chain-a"]
    # 链存在状态机违规不影响溯源返回，违规只体现在 validation 中
    chain_a_result = next(c for c in resp.chains if c.chain_id == "chain-a")
    assert chain_a_result.valid is False
    assert [v.index for v in chain_a_result.violations] == [2]


def test_source_chain_without_events_cannot_spread():
    resp = trace_contamination(
        make_request(
            chains=[
                chain("chain-a", []),
                chain("chain-b", [ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice")]),
            ],
            source_id="chain-a",
            contaminated_at=dt(8),
            audit_until=dt(20),
        )
    )
    # 污染源无持有区间：自身被污染但无法形成接触
    assert [item.chain_id for item in resp.contaminated] == ["chain-a"]
    assert resp.contaminated[0].path == ["chain-a"]
    assert resp.unreached == ["chain-b"]


# ---------------------------------------------------------------------------
# 422 语义错误
# ---------------------------------------------------------------------------


def test_unknown_source_raises_422():
    with pytest.raises(ContaminationTraceError, match="不存在于事件链中"):
        trace_contamination(
            make_request(
                chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
                source_id="ghost",
                contaminated_at=dt(9),
                audit_until=dt(20),
            )
        )


def test_audit_until_before_last_valid_event_raises_422():
    with pytest.raises(ContaminationTraceError, match="早于最后一个有效事件时间"):
        trace_contamination(
            make_request(
                chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
                source_id="chain-a",
                contaminated_at=dt(6),
                audit_until=dt(7),
            )
        )


def test_audit_until_before_contaminated_at_raises_422():
    with pytest.raises(ContaminationTraceError, match="早于污染源污染时刻"):
        trace_contamination(
            make_request(
                chains=[chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")])],
                source_id="chain-a",
                contaminated_at=dt(10),
                audit_until=dt(9, 30),
            )
        )


def test_mixed_timezone_raises_422():
    naive_event = Event(
        type=EventType.SEAL,
        timestamp=datetime.fromisoformat(f"{DAY}T09:00:00"),  # 不带时区
        seal_id="S2",
        custodian="Alice",
    )
    with pytest.raises(ContaminationTraceError, match="时区"):
        trace_contamination(
            make_request(
                chains=[
                    chain("chain-a", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
                    chain("chain-b", [naive_event]),
                ],
                source_id="chain-a",
                contaminated_at=dt(10),
                audit_until=dt(20),
            )
        )


def test_duplicate_chain_id_rejected():
    with pytest.raises(ValidationError):
        make_request(
            chains=[chain("chain-a", []), chain("chain-a", [])],
            source_id="chain-a",
            contaminated_at=dt(8),
            audit_until=dt(20),
        )


# ---------------------------------------------------------------------------
# API 集成
# ---------------------------------------------------------------------------


def test_api_contamination_trace_full_response():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
            {"chain_id": "chain-b", "events": [
                {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
                 "seal_id": "S002", "custodian": "Alice"},
                {"type": "handover", "timestamp": f"{DAY}T10:00:00Z",
                 "custodian": "Bob"},
            ]},
            {"chain_id": "chain-c", "events": [
                {"type": "seal", "timestamp": f"{DAY}T10:30:00Z",
                 "seal_id": "S003", "custodian": "Bob"},
            ]},
            {"chain_id": "chain-d", "events": [
                {"type": "seal", "timestamp": f"{DAY}T11:00:00Z",
                 "seal_id": "S004", "custodian": "Carol"},
            ]},
        ],
        "source": {"chain_id": "chain-a", "contaminated_at": f"{DAY}T09:30:00Z"},
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/contamination-trace", json=payload)
    assert r.status_code == 200
    body = r.json()

    assert body["audit_until"] == f"{DAY}T20:00:00Z"
    assert body["source"] == {"chain_id": "chain-a", "contaminated_at": f"{DAY}T09:30:00Z"}

    # 各链原校验结果保留（结构同单链接口并附 chain_id）
    assert [c["chain_id"] for c in body["chains"]] == [
        "chain-a", "chain-b", "chain-c", "chain-d",
    ]
    assert all(c["valid"] is True for c in body["chains"])
    assert body["chains"][1]["events"] == [
        {"index": 0, "type": "seal", "status": "ok", "reason": None,
         "cross_chain_conflict": False},
        {"index": 1, "type": "handover", "status": "ok", "reason": None,
         "cross_chain_conflict": False},
    ]

    assert body["contaminated"] == [
        {"chain_id": "chain-a", "earliest_contaminated_at": f"{DAY}T09:30:00Z",
         "direct_source": None, "path": ["chain-a"]},
        {"chain_id": "chain-b", "earliest_contaminated_at": f"{DAY}T09:30:00Z",
         "direct_source": "chain-a", "path": ["chain-a", "chain-b"]},
        {"chain_id": "chain-c", "earliest_contaminated_at": f"{DAY}T10:30:00Z",
         "direct_source": "chain-b", "path": ["chain-a", "chain-b", "chain-c"]},
    ]
    assert body["unreached"] == ["chain-d"]


def test_api_unknown_source_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
        ],
        "source": {"chain_id": "ghost", "contaminated_at": f"{DAY}T09:00:00Z"},
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/contamination-trace", json=payload)
    assert r.status_code == 422
    assert "不存在于事件链中" in r.json()["detail"]


def test_api_audit_until_too_early_returns_422():
    payload = {
        "chains": [
            {"chain_id": "chain-a", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S001", "custodian": "Alice"},
            ]},
        ],
        "source": {"chain_id": "chain-a", "contaminated_at": f"{DAY}T09:00:00Z"},
        "audit_until": f"{DAY}T07:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/contamination-trace", json=payload)
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
        "source": {"chain_id": "chain-a", "contaminated_at": f"{DAY}T09:00:00"},
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/contamination-trace", json=payload)
    assert r.status_code == 422
    assert "时区" in r.json()["detail"]


def test_api_duplicate_chain_id_returns_422():
    payload = {
        "chains": [{"chain_id": "chain-a", "events": []},
                   {"chain_id": "chain-a", "events": []}],
        "source": {"chain_id": "chain-a", "contaminated_at": f"{DAY}T08:00:00Z"},
        "audit_until": f"{DAY}T20:00:00Z",
    }
    r = client.post("/api/v1/seal-chain/contamination-trace", json=payload)
    assert r.status_code == 422


def test_api_missing_required_fields_returns_422():
    r = client.post("/api/v1/seal-chain/contamination-trace", json={"chains": []})
    assert r.status_code == 422
