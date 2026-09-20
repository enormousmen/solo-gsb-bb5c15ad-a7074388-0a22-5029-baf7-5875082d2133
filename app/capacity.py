"""保管容量审计。

处理流程：
1. 复用现有单链校验器（``validate_events``）逐链校验，各链原校验结果原样保留；
2. 仅以各链 ``status=ok`` 的 seal / reseal / handover 事件生成截至
   ``audit_until`` 的保管人半开持有区间 ``[start, end)``：每个有效事件以其
   时间戳开启对应保管人的区间，下一个有效事件闭合前一区间，末个区间向右延伸
   至 ``audit_until``；unseal 不改变保管人，违规事件一律不产生区间；
3. 按保管人扫描各自的区间起止边界：同刻事件**先处理结束、后处理开始**（半开
   区间语义，恰好在边界交接的两条链在该刻不并发）。相邻边界之间活跃链集合
   恒定，并发数超过该保管人容量的部分即为一个原始超限段；
4. 合并相邻且活跃链集合完全相同的超限段（边界处集合若未变化则不切分），
   返回每段起止、容量、并发数与按字典序排列的 ``chain_ids``；
5. 按保管人汇总超限总秒数与超限段内的并发峰值。

任何产生非空持有区间的保管人都必须在 ``capacities`` 中给出正整数容量（缺少
容量或保管人重复返回 422，重复由请求模型拦截）；事件与 ``audit_until`` 的
时间不得混用带时区与不带时区的时间（否则 422）；``audit_until`` 不得早于
任一有效事件时间（否则 422）。
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
    """容量审计入参的语义错误（在路由层转换为 HTTP 422）。"""


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
class _RawOverrun:
    """扫描得到的原始超限段：活跃链集合恒定且并发数超过容量。"""

    start: datetime
    end: datetime
    active: Set[str]  # 该段内活跃的 chain_id 集合


def _derive_holding_intervals(
    request: CustodyCapacityAuditRequest,
    chain_results: List[BatchChainResult],
) -> List[_HoldingInterval]:
    """从各链 status=ok 的 seal/reseal/handover 事件生成截至 audit_until 的
    保管人半开持有区间。

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


def _sweep_custodian(
    intervals: List[_HoldingInterval], capacity: int
) -> List[_RawOverrun]:
    """扫描单个保管人的全部持有区间，返回原始超限段（尚未合并）。

    边界事件按 (时刻, 同刻先结束后开始) 排序：结束事件标记为 0、开始事件
    标记为 1，恰好在同一刻交接的两条链在该刻不并发（半开区间语义）。
    同一时刻的事件全部应用完毕后，以更新后的活跃集合开一段到下一时刻。
    """
    events: List[Tuple[datetime, int, str]] = []
    for interval in intervals:
        events.append((interval.end, 0, interval.chain_id))    # 先处理结束
        events.append((interval.start, 1, interval.chain_id))  # 后处理开始
    events.sort(key=lambda item: (item[0], item[1]))

    active: Set[str] = set()
    raw: List[_RawOverrun] = []
    segment_start: Optional[datetime] = None

    cursor = 0
    while cursor < len(events):
        moment = events[cursor][0]
        if segment_start is not None and moment > segment_start and len(active) > capacity:
            raw.append(_RawOverrun(start=segment_start, end=moment, active=set(active)))
        # 应用该时刻的全部事件（先结束、后开始）
        while cursor < len(events) and events[cursor][0] == moment:
            _, kind, chain_id = events[cursor]
            if kind == 0:
                active.discard(chain_id)
            else:
                active.add(chain_id)
            cursor += 1
        segment_start = moment
    return raw


def _merge_raw_overruns(raw: List[_RawOverrun]) -> List[_RawOverrun]:
    """合并相邻且活跃链集合完全相同的原始超限段。

    边界两侧活跃集合相同意味着该边界不改变并发构成，两段应并为一段。
    """
    merged: List[_RawOverrun] = []
    for piece in raw:
        if merged and merged[-1].end == piece.start and merged[-1].active == piece.active:
            merged[-1] = _RawOverrun(
                start=merged[-1].start, end=piece.end, active=merged[-1].active
            )
        else:
            merged.append(piece)
    return merged


def audit_custody_capacity(
    request: CustodyCapacityAuditRequest,
) -> CustodyCapacityAuditResponse:
    """逐链校验并扫描各保管人并发容量，返回超限分段与按保管人汇总。"""
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

    # 阶段 3：audit_until 不得早于任一有效事件时间（违规事件不推进时间基准）
    last_valid: Optional[datetime] = None
    for chain, result in zip(request.chains, chain_results):
        for event, event_result in zip(chain.events, result.events):
            if event_result.status != "ok":
                continue
            if last_valid is None or event.timestamp > last_valid:
                last_valid = event.timestamp
    if last_valid is not None and request.audit_until < last_valid:
        raise CustodyCapacityAuditError(
            "audit_until 早于最后一个有效事件时间："
            f"{request.audit_until.isoformat()} 早于 {last_valid.isoformat()}"
        )

    # 阶段 4：生成截至 audit_until 的保管人持有区间
    intervals = _derive_holding_intervals(request, chain_results)

    by_custodian: Dict[str, List[_HoldingInterval]] = {}
    for interval in intervals:
        by_custodian.setdefault(interval.custodian, []).append(interval)

    capacity_map: Dict[str, int] = {
        item.custodian: item.capacity for item in request.capacities
    }

    # 阶段 5：任何产生非空持有区间的保管人都必须配置容量
    missing = sorted(name for name in by_custodian if name not in capacity_map)
    if missing:
        raise CustodyCapacityAuditError(f"缺少保管人容量配置：{', '.join(missing)}")

    # 阶段 6：按保管人扫描边界 → 原始超限段 → 合并相邻同集合段
    segments: List[CapacityOverrunSegment] = []
    summaries: List[CustodianCapacitySummary] = []
    for custodian in sorted(by_custodian):
        group = sorted(by_custodian[custodian], key=lambda iv: (iv.start, iv.chain_id))
        raw = _sweep_custodian(group, capacity_map[custodian])
        merged = _merge_raw_overruns(raw)
        if not merged:
            continue
        peak = 0
        total_seconds = 0.0
        for piece in merged:
            seconds = (piece.end - piece.start).total_seconds()
            concurrency = len(piece.active)
            segments.append(
                CapacityOverrunSegment(
                    custodian=custodian,
                    start=piece.start,
                    end=piece.end,
                    seconds=seconds,
                    capacity=capacity_map[custodian],
                    concurrency=concurrency,
                    chain_ids=sorted(piece.active),
                )
            )
            total_seconds += seconds
            if concurrency > peak:
                peak = concurrency
        summaries.append(
            CustodianCapacitySummary(
                custodian=custodian,
                capacity=capacity_map[custodian],
                overrun_seconds=total_seconds,
                peak_concurrency=peak,
            )
        )

    return CustodyCapacityAuditResponse(
        audit_until=request.audit_until,
        chains=chain_results,
        segments=segments,
        custodians=summaries,
    )
