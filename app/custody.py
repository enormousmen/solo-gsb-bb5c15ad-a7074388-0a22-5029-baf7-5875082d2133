"""保管容量并发审计。

处理流程：
1. 复用现有单链校验器（``validate_events``）逐链校验，各链原校验结果原样保留；
2. 仅以各链 ``status=ok`` 的 seal / reseal / handover 事件生成保管人半开持有
   区间 ``[start, end)``：每个有效事件以其时间戳开启对应保管人的区间，下一个
   有效事件闭合前一区间，末个区间向右延伸至 ``audit_until``；unseal 不改变
   保管人，违规事件一律不产生区间；
3. 按保管人收集其名下全部区间，以区间起止为边界做扫描线：同一时刻先处理
   结束（end）再处理开始（start）——半开区间语义下同刻交接不产生并发；
   相邻边界之间的切片对应一个恒定的活跃链集合；
4. 活跃链数超过容量的切片为超限切片；相邻（首尾相接）且活跃链集合完全相同
   的超限切片合并为一段，返回起止、容量、并发数与按字典序排列的 chain_ids；
5. 按保管人汇总超限累计秒数与审计窗口内的并发峰值。

事件与 ``audit_until`` 的时间不得混用带时区与不带时区的时间（否则 422）；
容量缺失（审计窗口内实际持有过链的保管人未给出容量）、容量重复、或
``audit_until`` 早于任一链最后一个有效事件时间时返回 422。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    BatchChainResult,
    CapacityOverrunSegment,
    CustodianCapacitySummary,
    CustodyCapacityAuditRequest,
    CustodyCapacityAuditResponse,
    EventType,
)
from .timeline import ensure_comparable, ok_event_records
from .validator import validate_events


class CustodyCapacityAuditError(ValueError):
    """审计入参的语义错误（在路由层转换为 HTTP 422）。"""


# 生成保管人持有区间的事件类型：seal / reseal / handover（unseal 不改变保管人）
_CUSTODIAN_TYPES = (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)


@dataclass(frozen=True)
class _HoldingInterval:
    """某条链在某保管人名下的一段半开持有区间 [start, end)。"""

    chain_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class _Slice:
    """扫描线在相邻边界之间切出的恒定活跃链集合片段 [start, end)。"""

    start: datetime
    end: datetime
    active: Set[str]


def _derive_holding_intervals(
    request: CustodyCapacityAuditRequest,
    chain_results: List[BatchChainResult],
) -> Dict[str, List[_HoldingInterval]]:
    """从各链 status=ok 的 seal/reseal/handover 事件生成保管人半开持有区间。

    每个有效事件开启对应保管人的区间，下一个有效事件闭合前一区间；
    末个区间向右延伸至 audit_until；与 audit_until 重合的零长度区间丢弃。
    返回按保管人分组的持有区间。
    """
    by_custodian: Dict[str, List[_HoldingInterval]] = {}
    for chain, result in zip(request.chains, chain_results):
        records = ok_event_records(chain.events, result.events, _CUSTODIAN_TYPES)
        for i, record in enumerate(records):
            end = records[i + 1].timestamp if i + 1 < len(records) else request.audit_until
            if record.timestamp < end:
                by_custodian.setdefault(record.custodian, []).append(  # type: ignore[arg-type]
                    _HoldingInterval(
                        chain_id=chain.chain_id,
                        start=record.timestamp,
                        end=end,
                    )
                )
    return by_custodian


def _sweep_slices(intervals: List[_HoldingInterval]) -> List[_Slice]:
    """按区间起止边界扫描，切出活跃链集合恒定的半开片段。

    同一时刻先处理结束（end）再处理开始（start）：半开区间语义下
    [.,t) 与 [t,.) 在 t 点不重叠，同刻交接不计并发。
    """
    # (时刻, 类型) 排序：0 = 结束（先处理），1 = 开始（后处理）
    events: List[Tuple[datetime, int, str]] = []
    for interval in intervals:
        events.append((interval.start, 1, interval.chain_id))  # 开始
        events.append((interval.end, 0, interval.chain_id))    # 结束
    events.sort(key=lambda item: (item[0], item[1]))

    active: Set[str] = set()
    slices: List[_Slice] = []
    current_start: Optional[datetime] = None
    for moment, kind, chain_id in events:
        if current_start is not None and moment > current_start and active:
            slices.append(_Slice(start=current_start, end=moment, active=set(active)))
        if kind == 0:
            active.discard(chain_id)
        else:
            active.add(chain_id)
        current_start = moment
    return slices


def _merge_overrun_segments(
    custodian: str,
    capacity: int,
    slices: List[_Slice],
) -> Tuple[List[CapacityOverrunSegment], int]:
    """把并发数超过容量的切片合并为超限段，返回（超限段, 并发峰值）。

    相邻（前一段 end == 后一段 start）且活跃链集合完全相同的超限切片
    合并为一段；中间夹着未超限切片、或活跃集合不同则不合并。
    """
    segments: List[CapacityOverrunSegment] = []
    peak = 0
    current: Optional[_Slice] = None
    for slice_ in slices:
        peak = max(peak, len(slice_.active))
        if len(slice_.active) <= capacity:
            if current is not None:
                segments.append(_make_segment(custodian, capacity, current))
                current = None
            continue
        if current is not None and current.end == slice_.start and current.active == slice_.active:
            # 相邻且活跃链集合相同：延伸当前超限段
            current = _Slice(start=current.start, end=slice_.end, active=current.active)
        else:
            if current is not None:
                segments.append(_make_segment(custodian, capacity, current))
            current = slice_
    if current is not None:
        segments.append(_make_segment(custodian, capacity, current))
    return segments, peak


def _make_segment(
    custodian: str, capacity: int, slice_: _Slice
) -> CapacityOverrunSegment:
    return CapacityOverrunSegment(
        custodian=custodian,
        start=slice_.start,
        end=slice_.end,
        seconds=(slice_.end - slice_.start).total_seconds(),
        capacity=capacity,
        concurrency=len(slice_.active),
        chain_ids=sorted(slice_.active),
    )


def audit_custody_capacity(
    request: CustodyCapacityAuditRequest,
) -> CustodyCapacityAuditResponse:
    """逐链校验并按保管人审计持有并发是否超出容量。"""
    # 阶段 1：逐链复用单链校验，结果原样保留
    chain_results: List[BatchChainResult] = []
    for chain in request.chains:
        single = validate_events(chain.events)
        chain_results.append(BatchChainResult(chain_id=chain.chain_id, **single.model_dump()))

    # 阶段 2：事件与 audit_until 不得混用时区
    timestamps = [request.audit_until]
    for chain in request.chains:
        timestamps.extend(event.timestamp for event in chain.events)
    ensure_comparable(timestamps, CustodyCapacityAuditError)

    # 阶段 3：容量 custodian 不可重复
    capacities: Dict[str, int] = {}
    for item in request.capacities:
        if item.custodian in capacities:
            raise CustodyCapacityAuditError(
                f"保管人容量重复：{item.custodian}"
            )
        capacities[item.custodian] = item.capacity

    # 阶段 4：audit_until 不得早于任一链最后一个有效事件时间
    last_valid: Optional[datetime] = None
    for chain, result in zip(request.chains, chain_results):
        for event, event_result in zip(chain.events, result.events):
            if event_result.status != "ok":
                continue  # 违规事件不推进有效时间基准
            if last_valid is None or event.timestamp > last_valid:
                last_valid = event.timestamp
    if last_valid is not None and request.audit_until < last_valid:
        raise CustodyCapacityAuditError(
            "audit_until 早于最后一个有效事件时间："
            f"{request.audit_until.isoformat()} 早于 {last_valid.isoformat()}"
        )

    # 阶段 5：生成持有区间并按保管人分组
    by_custodian = _derive_holding_intervals(request, chain_results)

    # 阶段 6：审计窗口内实际持有过链的保管人均须给出容量
    missing = sorted(custodian for custodian in by_custodian if custodian not in capacities)
    if missing:
        raise CustodyCapacityAuditError(
            "缺少保管人容量：" + "、".join(missing)
        )

    # 阶段 7：逐保管人扫描、合并超限段并汇总
    segments: List[CapacityOverrunSegment] = []
    summaries: List[CustodianCapacitySummary] = []
    for custodian in sorted(by_custodian):  # 字典序：汇总与分段均确定
        custodian_segments, peak = _merge_overrun_segments(
            custodian,
            capacities[custodian],
            _sweep_slices(by_custodian[custodian]),
        )
        segments.extend(custodian_segments)  # 保管人已按字典序，段内按时间升序
        summaries.append(
            CustodianCapacitySummary(
                custodian=custodian,
                capacity=capacities[custodian],
                overrun_seconds=sum(segment.seconds for segment in custodian_segments),
                peak_concurrency=peak,
            )
        )

    return CustodyCapacityAuditResponse(
        audit_until=request.audit_until,
        chains=chain_results,
        segments=segments,
        custodians=summaries,
    )
