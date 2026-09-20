"""污染传播溯源。

处理流程：
1. 复用现有单链校验器（``validate_events``）逐链校验，各链原校验结果原样保留；
2. 仅以各链 ``status=ok`` 的 seal / reseal / handover 事件生成保管人半开持有
   区间 ``[start, end)``：每个有效事件以其时间戳开启对应保管人的区间，下一个
   有效事件闭合前一区间，末个区间向右延伸至 ``audit_until``；unseal 不改变
   保管人，违规事件一律不产生区间；
3. 两条链的展品在同一保管人名下区间重叠（半开区间相交）即形成一次接触，
   接触窗口为重叠部分 ``[max(start), min(end))``；
4. 污染源链自 ``contaminated_at`` 起被污染；污染只能沿接触向前传播：若链 A
   在 ``tA`` 已被污染，且与链 B 存在接触窗口 ``[c1, c2)``，则 B 在
   ``max(tA, c1)`` 时刻被污染——前提是该时刻接触仍在持续（``< c2``）。
   按 (污染时刻, chain_id) 时间优先队列级联计算每条链的最早污染时刻；
5. 直接来源在级联定序后回放确定：在已定序且能在该链最早污染时刻送达污染的
   链中取 ``chain_id`` 字典序最小者（同一最早时刻并列时据此固定结果），
   完整传播路径沿直接来源回溯至污染源；
6. 审计截止前未被污染到达的链单独列入 ``unreached``。

事件、污染源与 ``audit_until`` 的时间不得混用带时区与不带时区的时间（否则
422）；污染源 ``chain_id`` 必须出现在事件链中（否则 422）；``audit_until``
不得早于任一有效事件时间或 ``contaminated_at``（否则 422）。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import (
    BatchChainResult,
    ContaminatedChainResult,
    ContaminationTraceRequest,
    ContaminationTraceResponse,
    EventType,
)
from .timeline import ensure_comparable, ok_event_records
from .validator import validate_events


class ContaminationTraceError(ValueError):
    """溯源入参的语义错误（在路由层转换为 HTTP 422）。"""


# 生成保管人持有区间的事件类型：seal / reseal / handover（unseal 不改变保管人）
_CUSTODIAN_TYPES = (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)


@dataclass(frozen=True)
class _HoldingInterval:
    """一段保管人半开持有区间 [start, end)（已按 audit_until 截断）。"""

    chain_id: str
    custodian: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class _Contact:
    """一次接触：两条链在同一保管人名下的持有区间重叠窗口 [start, end)。"""

    chain_a: str
    chain_b: str
    start: datetime
    end: datetime


def _derive_holding_intervals(
    request: ContaminationTraceRequest,
    chain_results: List[BatchChainResult],
) -> List[_HoldingInterval]:
    """从各链 status=ok 的 seal/reseal/handover 事件生成保管人半开持有区间。

    每个有效事件开启对应保管人的区间，下一个有效事件闭合前一区间；
    末个区间向右延伸至 audit_until；与 audit_until 重合的零长度区间丢弃。
    """
    intervals: List[_HoldingInterval] = []
    for chain, result in zip(request.chains, chain_results):
        records = ok_event_records(chain.events, result.events, _CUSTODIAN_TYPES)
        for i, record in enumerate(records):
            if i + 1 < len(records):
                end = records[i + 1].timestamp
            else:
                end = request.audit_until  # 末个区间延伸至审计截止时间
            if record.timestamp < end:
                intervals.append(
                    _HoldingInterval(
                        chain_id=chain.chain_id,
                        custodian=record.custodian,  # type: ignore[arg-type]
                        start=record.timestamp,
                        end=end,
                    )
                )
    return intervals


def _derive_contacts(intervals: List[_HoldingInterval]) -> List[_Contact]:
    """同一保管人名下持有区间重叠（跨链）即形成接触，接触窗口为重叠部分。"""
    by_custodian: Dict[str, List[_HoldingInterval]] = {}
    for interval in intervals:
        by_custodian.setdefault(interval.custodian, []).append(interval)

    contacts: List[_Contact] = []
    for custodian in sorted(by_custodian):  # 确定性的接触生成顺序
        group = sorted(by_custodian[custodian], key=lambda iv: (iv.start, iv.chain_id))
        active: List[_HoldingInterval] = []
        for interval in group:
            active = [iv for iv in active if iv.end > interval.start]
            for other in active:
                if other.chain_id == interval.chain_id:
                    continue  # 同链区间互不重叠，防御性跳过
                # 已按 start 升序，重叠窗口起点即 interval.start
                contacts.append(
                    _Contact(
                        chain_a=other.chain_id,
                        chain_b=interval.chain_id,
                        start=interval.start,
                        end=min(other.end, interval.end),
                    )
                )
            active.append(interval)
    return contacts


def _cascade_contamination(
    chain_ids: List[str],
    contacts: List[_Contact],
    source_id: str,
    contaminated_at: datetime,
) -> Tuple[Dict[str, datetime], Dict[str, Optional[str]]]:
    """级联计算各链最早污染时刻与直接来源。

    时间优先队列按 (污染时刻, chain_id) 出队：候选污染时刻为
    ``max(源链污染时刻, 接触起点)``，且须早于接触终点（接触仍在持续）。
    直接来源在定序后回放：在已定序且能在该链最早污染时刻送达的链中取
    chain_id 字典序最小者；级联松弛保证每条到达链至少存在一个这样的送达者，
    且送达者必然先于本链定序（沿直接来源回溯必然终止于污染源，无环）。
    """
    adjacency: Dict[str, List[Tuple[str, datetime, datetime]]] = {
        chain_id: [] for chain_id in chain_ids
    }
    for contact in contacts:
        adjacency[contact.chain_a].append((contact.chain_b, contact.start, contact.end))
        adjacency[contact.chain_b].append((contact.chain_a, contact.start, contact.end))

    best: Dict[str, datetime] = {source_id: contaminated_at}
    heap: List[Tuple[datetime, str]] = [(contaminated_at, source_id)]
    finalized: Dict[str, datetime] = {}
    order: List[str] = []
    while heap:
        moment, chain_id = heapq.heappop(heap)
        if chain_id in finalized:
            continue
        finalized[chain_id] = moment
        order.append(chain_id)
        for other, contact_start, contact_end in adjacency[chain_id]:
            if other in finalized:
                continue
            candidate = max(moment, contact_start)
            if candidate >= contact_end:
                continue  # 接触在源链被污染前已结束，无法传播
            if other not in best or candidate < best[other]:
                best[other] = candidate
                heapq.heappush(heap, (candidate, other))

    direct: Dict[str, Optional[str]] = {source_id: None}
    done = {source_id}
    for chain_id in order[1:]:
        moment = finalized[chain_id]
        deliverers = set()
        for other, contact_start, contact_end in adjacency[chain_id]:
            if other not in done:
                continue
            arrival = max(finalized[other], contact_start)
            if arrival == moment and arrival < contact_end:
                deliverers.add(other)
        # 同一最早时刻并列时按 chain_id 字典序固定直接来源
        direct[chain_id] = min(deliverers)
        done.add(chain_id)
    return finalized, direct


def _build_path(chain_id: str, direct: Dict[str, Optional[str]]) -> List[str]:
    """沿直接来源回溯至污染源，返回从污染源到本链的完整传播路径。"""
    path = [chain_id]
    while True:
        parent = direct[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path


def trace_contamination(request: ContaminationTraceRequest) -> ContaminationTraceResponse:
    """逐链校验并级联计算污染传播，返回各链最早污染时刻与传播路径。"""
    # 阶段 1：逐链复用单链校验，结果原样保留
    chain_results: List[BatchChainResult] = []
    for chain in request.chains:
        single = validate_events(chain.events)
        chain_results.append(BatchChainResult(chain_id=chain.chain_id, **single.model_dump()))

    # 阶段 2：事件、污染源与 audit_until 不得混用时区
    timestamps = [request.source.contaminated_at, request.audit_until]
    for chain in request.chains:
        timestamps.extend(event.timestamp for event in chain.events)
    ensure_comparable(timestamps, ContaminationTraceError)

    # 阶段 3：污染源 chain_id 必须出现在事件链中
    chain_ids = [chain.chain_id for chain in request.chains]
    if request.source.chain_id not in chain_ids:
        raise ContaminationTraceError(
            f"污染源 chain_id 不存在于事件链中：{request.source.chain_id}"
        )

    # 阶段 4：audit_until 不得早于任一有效事件时间或 contaminated_at
    last_valid: Optional[datetime] = None
    for chain, result in zip(request.chains, chain_results):
        for event, event_result in zip(chain.events, result.events):
            if event_result.status != "ok":
                continue  # 违规事件不推进有效时间基准
            if last_valid is None or event.timestamp > last_valid:
                last_valid = event.timestamp
    if last_valid is not None and request.audit_until < last_valid:
        raise ContaminationTraceError(
            "audit_until 早于最后一个有效事件时间："
            f"{request.audit_until.isoformat()} 早于 {last_valid.isoformat()}"
        )
    if request.audit_until < request.source.contaminated_at:
        raise ContaminationTraceError(
            "audit_until 早于污染源污染时刻："
            f"{request.audit_until.isoformat()} 早于 "
            f"{request.source.contaminated_at.isoformat()}"
        )

    # 阶段 5：持有区间 → 接触 → 级联溯源
    intervals = _derive_holding_intervals(request, chain_results)
    contacts = _derive_contacts(intervals)
    times, direct = _cascade_contamination(
        chain_ids, contacts, request.source.chain_id, request.source.contaminated_at
    )

    contaminated = [
        ContaminatedChainResult(
            chain_id=chain_id,
            earliest_contaminated_at=times[chain_id],
            direct_source=direct[chain_id],
            path=_build_path(chain_id, direct),
        )
        for chain_id in sorted(times, key=lambda cid: (times[cid], cid))
    ]
    unreached = sorted(chain_id for chain_id in chain_ids if chain_id not in times)

    return ContaminationTraceResponse(
        audit_until=request.audit_until,
        source=request.source,
        chains=chain_results,
        contaminated=contaminated,
        unreached=unreached,
    )
