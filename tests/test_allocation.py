"""封签库存分配规划的单元测试与 API 集成测试。"""
from __future__ import annotations

import random
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.allocation import AllocationPlanError, _optimal_allocation, plan_allocation
from app.main import app
from app.models import (
    AllocationPlanRequest,
    ChainEvents,
    Event,
    EventType,
    InventorySeal,
    PendingUse,
)

client = TestClient(app)

DAY = "2026-09-19"
USE_DAY = "2026-09-20"


def dt(hour: int, minute: int = 0, day: str = DAY) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def chain(chain_id: str, events) -> ChainEvents:
    return ChainEvents(chain_id=chain_id, events=list(events))


def rec(chain_id: str, custodian: str, hour: int, minute: int = 0) -> PendingUse:
    return PendingUse(chain_id=chain_id, custodian=custodian, use_at=dt(hour, minute, USE_DAY))


def seal(seal_id: str, hour: int, minute: int = 0, allowed=("Alice",)) -> InventorySeal:
    return InventorySeal(
        seal_id=seal_id,
        expires_at=dt(hour, minute, USE_DAY),
        allowed_custodians=list(allowed),
    )


def make_request(chains=(), pending=(), inventory=()) -> AllocationPlanRequest:
    return AllocationPlanRequest(
        chains=list(chains), pending=list(pending), inventory=list(inventory)
    )


def record_by_index(resp):
    return {r.index: r for r in resp.records}


# ---------------------------------------------------------------------------
# 基本分配与候选边规则（未过期 + 保管人允许）
# ---------------------------------------------------------------------------


def test_simple_assignment():
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    assert resp.records[0].status == "assigned"
    assert resp.records[0].seal_id == "S1"
    assert resp.records[0].reason is None
    assert resp.summary.assigned == 1
    assert resp.summary.total_slack_seconds == 3 * 3600.0


def test_expired_seal_is_not_a_candidate():
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 10)],
            inventory=[seal("S1", 9)],  # 09:00 过期，use_at 10:00 已过期
        )
    )
    assert resp.records[0].status == "unassigned"
    assert resp.records[0].reason == "no_candidate"
    assert resp.records[0].seal_id is None
    assert resp.summary.no_candidate == 1


def test_use_at_equal_expires_at_is_expired():
    # use_at 恰好等于 expires_at：恰在过期时刻使用视为已过期
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 10)],
            inventory=[seal("S1", 10)],
        )
    )
    assert resp.records[0].status == "unassigned"
    assert resp.records[0].reason == "no_candidate"


def test_custodian_not_allowed_is_not_a_candidate():
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Bob", 9)],
            inventory=[seal("S1", 12, allowed=("Alice",))],
        )
    )
    assert resp.records[0].status == "unassigned"
    assert resp.records[0].reason == "no_candidate"


def test_empty_allowed_custodians_allows_nobody():
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12, allowed=())],
        )
    )
    assert resp.records[0].reason == "no_candidate"


def test_one_seal_serves_at_most_one_record():
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9), rec("chain-b", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    statuses = sorted(r.status for r in resp.records)
    assert statuses == ["assigned", "unassigned"]
    assert resp.summary.assigned == 1
    assert resp.summary.outcompeted == 1


# ---------------------------------------------------------------------------
# 复用批量校验排除已占用编号
# ---------------------------------------------------------------------------


def test_occupied_seal_id_is_excluded():
    resp = plan_allocation(
        make_request(
            chains=[chain("chain-a", [ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")])],
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12), seal("S2", 12)],
        )
    )
    assert resp.excluded_seal_ids == ["S1"]
    assert resp.records[0].status == "assigned"
    assert resp.records[0].seal_id == "S2"
    assert resp.batch_validation.valid is True


def test_all_inventory_occupied_leaves_no_candidate():
    resp = plan_allocation(
        make_request(
            chains=[chain("chain-a", [ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")])],
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    assert resp.excluded_seal_ids == ["S1"]
    assert resp.records[0].status == "unassigned"
    assert resp.records[0].reason == "no_candidate"


def test_seal_in_violation_event_is_not_occupied():
    # 未封装状态下的 reseal 违规：其中的 seal_id 不构成占用
    resp = plan_allocation(
        make_request(
            chains=[chain("chain-a", [ev(EventType.RESEAL, 10, seal_id="S1", custodian="Alice")])],
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    assert resp.batch_validation.valid is False
    assert resp.excluded_seal_ids == []
    assert resp.records[0].status == "assigned"
    assert resp.records[0].seal_id == "S1"


def test_cross_chain_conflict_still_occupies():
    # 跨链冲突不改变事件有效性：两条链的有效 seal 都占用同一编号
    resp = plan_allocation(
        make_request(
            chains=[
                chain("chain-a", [ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")]),
                chain("chain-b", [ev(EventType.SEAL, 11, seal_id="S1", custodian="Bob")]),
            ],
            pending=[rec("chain-c", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    assert resp.batch_validation.valid is False  # 存在跨链冲突
    assert len(resp.batch_validation.conflicts) == 2
    assert resp.excluded_seal_ids == ["S1"]
    assert resp.records[0].reason == "no_candidate"


def test_reseal_occupies_and_excluded_ids_sorted():
    resp = plan_allocation(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.SEAL, 10, seal_id="S2", custodian="Alice"),
                        ev(EventType.UNSEAL, 11),
                        ev(EventType.RESEAL, 12, seal_id="S1", custodian="Alice"),
                    ],
                )
            ],
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 13), seal("S2", 13), seal("S3", 13)],
        )
    )
    assert resp.excluded_seal_ids == ["S1", "S2"]
    assert resp.records[0].seal_id == "S3"


# ---------------------------------------------------------------------------
# 全局最优：最大化满足数优先（逐条贪心会失败的场景）
# ---------------------------------------------------------------------------


def test_maximize_satisfied_count_beats_greedy():
    # r0(Alice) 候选 {S1, S2} 且 S2 余量更小；r1(Bob) 候选 {S2}。
    # “每条记录取余量最小可用封签”的逐条贪心：r0←S2 → r1 无解（满足 1 条）；
    # 全局最优：r0←S1、r1←S2，满足 2 条。
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9), rec("chain-b", "Bob", 9)],
            inventory=[
                seal("S1", 12, allowed=("Alice",)),
                seal("S2", 11, allowed=("Alice", "Bob")),
            ],
        )
    )
    results = record_by_index(resp)
    assert results[0].status == "assigned" and results[0].seal_id == "S1"
    assert results[1].status == "assigned" and results[1].seal_id == "S2"
    assert resp.summary.assigned == 2


# ---------------------------------------------------------------------------
# 全局最优：满足数相同时总秒数最小
# ---------------------------------------------------------------------------


def test_min_total_slack_is_global():
    # r0 候选 {S1, S2}，r1 候选 {S1, S3}；use_at 均 09:00，
    # S1/S2/S3 分别 11:00/11:30/12:00 过期（余量 7200/9000/10800 秒）。
    # 满足 2 条的方案中 r0←S2+r1←S1 总秒数最小（16200）；
    # 字典序更小的 r0←S1+r1←S3（18000）必须让位于总秒数目标。
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9), rec("chain-b", "Bob", 9)],
            inventory=[
                seal("S1", 11, allowed=("Alice", "Bob")),
                seal("S2", 11, 30, allowed=("Alice",)),
                seal("S3", 12, allowed=("Bob",)),
            ],
        )
    )
    results = record_by_index(resp)
    assert results[0].seal_id == "S2"
    assert results[1].seal_id == "S1"
    assert resp.summary.total_slack_seconds == 9000.0 + 7200.0


def test_min_total_slack_selects_earliest_expiring_set():
    # r2 无候选；r0/r1 与三枚封签全连接。满足 2 条时选余量总和最小的
    # 封签集合 {S1, S2}，字典序定 r0←S1、r1←S2。
    resp = plan_allocation(
        make_request(
            pending=[
                rec("chain-a", "Alice", 9),
                rec("chain-b", "Alice", 9),
                rec("chain-c", "Carol", 9),
            ],
            inventory=[seal("S3", 12), seal("S1", 11), seal("S2", 11, 30)],
        )
    )
    results = record_by_index(resp)
    assert results[0].seal_id == "S1"
    assert results[1].seal_id == "S2"
    assert results[2].status == "unassigned"
    assert results[2].reason == "no_candidate"
    assert resp.summary.total_slack_seconds == 7200.0 + 9000.0


def test_min_total_slack_prefers_earlier_expiring_set():
    # 满足 1 条的方案中，用 S1（余量 3600s）优于用 S2（余量 7200s）
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S2", 11), seal("S1", 10)],
        )
    )
    assert resp.records[0].seal_id == "S1"
    assert resp.summary.total_slack_seconds == 3600.0


# ---------------------------------------------------------------------------
# 全局最优：仍并列时按记录输入顺序与 seal_id 字典序定唯一方案
# ---------------------------------------------------------------------------


def test_lexicographic_seal_choice_on_tie():
    # 两条记录、两枚封签余量全同：r0←S1、r1←S2 的分配向量字典序最小
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9), rec("chain-b", "Alice", 9)],
            inventory=[seal("S2", 11), seal("S1", 11)],
        )
    )
    results = record_by_index(resp)
    assert results[0].seal_id == "S1"
    assert results[1].seal_id == "S2"


def test_earlier_record_wins_on_tie():
    # 一枚封签、两条记录余量相同：输入靠前的记录优先被满足
    resp = plan_allocation(
        make_request(
            pending=[rec("chain-a", "Alice", 9), rec("chain-b", "Alice", 9)],
            inventory=[seal("S1", 12)],
        )
    )
    results = record_by_index(resp)
    assert results[0].status == "assigned"
    assert results[1].status == "unassigned"
    assert results[1].reason == "outcompeted"


def test_outcompeted_reason_only_when_candidates_exist():
    resp = plan_allocation(
        make_request(
            pending=[
                rec("chain-a", "Alice", 9),   # 与下一条竞争同一枚 S1
                rec("chain-b", "Alice", 9),
                rec("chain-c", "Carol", 9),   # 无任何候选
            ],
            inventory=[seal("S1", 12, allowed=("Alice",))],
        )
    )
    results = record_by_index(resp)
    assert results[0].status == "assigned"
    assert results[1].reason == "outcompeted"
    assert results[2].reason == "no_candidate"
    assert resp.summary.model_dump() == {
        "assigned": 1,
        "unassigned": 2,
        "no_candidate": 1,
        "outcompeted": 1,
        "total_slack_seconds": 3 * 3600.0,
    }


# ---------------------------------------------------------------------------
# 空输入与批量校验结果保留
# ---------------------------------------------------------------------------


def test_empty_request():
    resp = plan_allocation(make_request())
    assert resp.batch_validation.valid is True
    assert resp.excluded_seal_ids == []
    assert resp.records == []
    assert resp.summary.model_dump() == {
        "assigned": 0,
        "unassigned": 0,
        "no_candidate": 0,
        "outcompeted": 0,
        "total_slack_seconds": 0.0,
    }


def test_batch_validation_result_is_preserved():
    resp = plan_allocation(
        make_request(
            chains=[
                chain(
                    "chain-a",
                    [
                        ev(EventType.HANDOVER, 9, custodian="Bob"),  # 违规：未封装交接
                        ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                    ],
                )
            ],
            pending=[rec("chain-a", "Alice", 9)],
            inventory=[seal("S1", 12), seal("S2", 12)],
        )
    )
    assert resp.batch_validation.valid is False
    assert [e.status for e in resp.batch_validation.chains[0].events] == [
        "violation",
        "ok",
    ]
    assert resp.excluded_seal_ids == ["S1"]
    assert resp.records[0].seal_id == "S2"


# ---------------------------------------------------------------------------
# 动态规划与暴力枚举对拍
# ---------------------------------------------------------------------------


def _brute_force_best(candidates, slacks, n, m):
    """递归枚举全部一对一分配，返回与 DP 相同口径的最优状态键。"""
    best = None
    choices = [m] * n
    used = [False] * m

    def search(i, neg_count, total):
        nonlocal best
        if i == n:
            key = (neg_count, total, tuple(choices))
            if best is None or key < best:
                best = key
            return
        choices[i] = m  # 记录 i 不分配
        search(i + 1, neg_count, total)
        for j in candidates[i]:
            if used[j]:
                continue
            used[j] = True
            choices[i] = j
            search(i + 1, neg_count - 1, total + slacks[i][j])
            used[j] = False
        choices[i] = m

    search(0, 0, 0.0)
    return best


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 5)
    m = rng.randint(0, 5)
    candidates = []
    slacks = []
    for _ in range(n):
        row = [j for j in range(m) if rng.random() < 0.6]
        candidates.append(row)
        slacks.append({j: float(rng.randint(0, 5000)) for j in row})

    assert _optimal_allocation(candidates, slacks, n, m) == _brute_force_best(
        candidates, slacks, n, m
    )


# ---------------------------------------------------------------------------
# 422 语义错误
# ---------------------------------------------------------------------------


def test_duplicate_inventory_seal_id_rejected():
    with pytest.raises(ValidationError):
        make_request(inventory=[seal("S1", 12), seal("S1", 13)])


def test_duplicate_chain_id_rejected():
    with pytest.raises(ValidationError):
        make_request(chains=[chain("chain-a", []), chain("chain-a", [])])


def test_empty_custodian_rejected():
    with pytest.raises(ValidationError):
        PendingUse(chain_id="chain-a", custodian="", use_at=dt(9, day=USE_DAY))


def test_mixed_timezone_raises_error():
    request = AllocationPlanRequest.model_construct(
        chains=[],
        pending=[rec("chain-a", "Alice", 9)],
        inventory=[
            InventorySeal.model_construct(
                seal_id="S1",
                expires_at=datetime.fromisoformat(f"{USE_DAY}T12:00:00"),  # naive
                allowed_custodians=["Alice"],
            )
        ],
    )
    with pytest.raises(AllocationPlanError):
        plan_allocation(request)


def test_mixed_timezone_between_events_and_pending_raises_error():
    request = AllocationPlanRequest.model_construct(
        chains=[
            ChainEvents.model_construct(
                chain_id="chain-a",
                events=[
                    Event.model_construct(
                        type=EventType.SEAL,
                        timestamp=datetime.fromisoformat(f"{DAY}T10:00:00"),  # naive
                        seal_id="S9",
                        custodian="Alice",
                    )
                ],
            )
        ],
        pending=[rec("chain-a", "Alice", 9)],  # aware
        inventory=[],
    )
    with pytest.raises(AllocationPlanError):
        plan_allocation(request)


# ---------------------------------------------------------------------------
# API 集成
# ---------------------------------------------------------------------------


def test_api_allocation_plan_full_response():
    payload = {
        "chains": [
            {
                "chain_id": "chain-a",
                "events": [
                    {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
                     "seal_id": "S001", "custodian": "Alice"}
                ],
            }
        ],
        "pending": [
            {"chain_id": "chain-a", "custodian": "Alice", "use_at": f"{USE_DAY}T09:00:00Z"},
            {"chain_id": "chain-b", "custodian": "Bob", "use_at": f"{USE_DAY}T09:30:00Z"},
            {"chain_id": "chain-c", "custodian": "Carol", "use_at": f"{USE_DAY}T10:00:00Z"},
        ],
        "inventory": [
            {"seal_id": "S001", "expires_at": f"{USE_DAY}T12:00:00Z",
             "allowed_custodians": ["Alice", "Bob"]},
            {"seal_id": "S002", "expires_at": f"{USE_DAY}T12:00:00Z",
             "allowed_custodians": ["Alice", "Bob"]},
            {"seal_id": "S003", "expires_at": f"{USE_DAY}T09:45:00Z",
             "allowed_custodians": ["Bob"]},
        ],
    }
    r = client.post("/api/v1/seal-chain/allocation-plan", json=payload)
    assert r.status_code == 200
    body = r.json()

    # S001 被 chain-a 有效 seal 占用而排除；Carol 无任何候选
    assert body["excluded_seal_ids"] == ["S001"]
    assert body["batch_validation"]["valid"] is True
    assert body["batch_validation"]["chains"][0]["chain_id"] == "chain-a"
    assert body["records"] == [
        {"index": 0, "chain_id": "chain-a", "custodian": "Alice",
         "use_at": f"{USE_DAY}T09:00:00Z",
         "status": "assigned", "seal_id": "S002", "reason": None},
        {"index": 1, "chain_id": "chain-b", "custodian": "Bob",
         "use_at": f"{USE_DAY}T09:30:00Z",
         "status": "assigned", "seal_id": "S003", "reason": None},
        {"index": 2, "chain_id": "chain-c", "custodian": "Carol",
         "use_at": f"{USE_DAY}T10:00:00Z",
         "status": "unassigned", "seal_id": None, "reason": "no_candidate"},
    ]
    assert body["summary"] == {
        "assigned": 2,
        "unassigned": 1,
        "no_candidate": 1,
        "outcompeted": 0,
        "total_slack_seconds": 10800.0 + 900.0,
    }


def test_api_empty_request_returns_200():
    r = client.post("/api/v1/seal-chain/allocation-plan", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["records"] == []
    assert body["excluded_seal_ids"] == []
    assert body["summary"]["assigned"] == 0


def test_api_duplicate_inventory_seal_id_returns_422():
    payload = {
        "inventory": [
            {"seal_id": "S1", "expires_at": f"{USE_DAY}T12:00:00Z",
             "allowed_custodians": ["Alice"]},
            {"seal_id": "S1", "expires_at": f"{USE_DAY}T13:00:00Z",
             "allowed_custodians": ["Alice"]},
        ]
    }
    r = client.post("/api/v1/seal-chain/allocation-plan", json=payload)
    assert r.status_code == 422


def test_api_duplicate_chain_id_returns_422():
    payload = {"chains": [{"chain_id": "chain-a"}, {"chain_id": "chain-a"}]}
    r = client.post("/api/v1/seal-chain/allocation-plan", json=payload)
    assert r.status_code == 422


def test_api_mixed_timezone_returns_422():
    payload = {
        "pending": [
            {"chain_id": "chain-a", "custodian": "Alice", "use_at": f"{USE_DAY}T09:00:00Z"}
        ],
        "inventory": [
            {"seal_id": "S1", "expires_at": f"{USE_DAY}T12:00:00",  # 不带时区
             "allowed_custodians": ["Alice"]}
        ],
    }
    r = client.post("/api/v1/seal-chain/allocation-plan", json=payload)
    assert r.status_code == 422
    assert "时区" in r.json()["detail"]


def test_api_existing_endpoints_unchanged():
    # 既有接口行为不受新接口影响
    r = client.post(
        "/api/v1/seal-chain/validate",
        json={
            "events": [
                {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
                 "seal_id": "S1", "custodian": "Alice"}
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["current_seal"] == "S1"
