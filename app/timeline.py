"""时间线公共逻辑。

开封暴露审计、交接计划符合性审计与温控责任审计共享的领域无关时间线工具：

1. ``ok_event_records``：按校验结果过滤 ``status=ok`` 且类型匹配的事件，
   保留全链事件下标，返回领域无关记录（类型 / 时间戳 / 保管人），
   由调用方映射为暴露区间、到任记录或保管人区间字段；
2. ``split_by_cover``：半开区间分割器，把目标半开区间 ``[start, end)`` 按
   有序覆盖区间切为 covered / gap 片段，支持相接边界（半开区间下
   ``end == next.start`` 的相接处被切开）与开放末端（覆盖区间
   ``end=None`` 表示向右开放）；
3. ``ensure_comparable``：统一时区可比性检查，混用带时区与不带时区的时间
   时报错；异常类型由调用方指定，以保持各接口既有 422 detail。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Iterable, List, Optional, Sequence, Type, TypeVar

from .models import Event, EventResult, EventType

#: 混用带时区与不带时区时间时的统一报错文案（各接口 422 detail 保持不变）
MIXED_TIMEZONE_MESSAGE = "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"


@dataclass(frozen=True)
class OkEventRecord:
    """一条 status=ok 事件的领域无关记录（index 为全链事件下标）。"""

    index: int
    type: EventType
    timestamp: datetime
    custodian: Optional[str]


def ok_event_records(
    events: Sequence[Event],
    event_results: Sequence[EventResult],
    event_types: Iterable[EventType],
) -> List[OkEventRecord]:
    """过滤 status=ok 且类型在 ``event_types`` 内的事件，保留全链下标。

    违规事件一律跳过；记录按事件顺序（下标升序）返回。领域无关：
    调用方自行把记录映射为暴露、到任或温控字段。
    """
    wanted = frozenset(event_types)
    records: List[OkEventRecord] = []
    for event, result in zip(events, event_results):
        if result.status != "ok" or event.type not in wanted:
            continue
        records.append(
            OkEventRecord(
                index=result.index,
                type=event.type,
                timestamp=event.timestamp,
                custodian=event.custodian,
            )
        )
    return records


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class CoverInterval(Generic[PayloadT]):
    """一段有序覆盖区间 [start, end)；end 为 None 表示向右开放。"""

    start: datetime
    end: Optional[datetime]
    payload: Optional[PayloadT] = None  # 调用方附属数据，随 covered 片段返回


@dataclass(frozen=True)
class TimelinePiece(Generic[PayloadT]):
    """目标区间被切出的一段：covered（命中覆盖区间）或 gap（未覆盖）。"""

    start: datetime
    end: datetime
    covered: bool
    payload: Optional[PayloadT] = None  # covered 时为覆盖区间的 payload；gap 为 None

    @property
    def seconds(self) -> float:
        """片段时长（秒），等于 end - start。"""
        return (self.end - self.start).total_seconds()


def split_by_cover(
    start: datetime,
    end: datetime,
    covers: Sequence[CoverInterval[PayloadT]],
) -> List[TimelinePiece[PayloadT]]:
    """把目标半开区间 [start, end) 按有序覆盖区间切为 covered / gap 片段。

    覆盖区间必须按起点升序且互不重叠（允许首尾相接）；返回片段按时间升序，
    相接边界处被切开，``end=None`` 的覆盖区间向右开放；退化目标区间
    （``start == end``）不产生任何片段。
    """
    pieces: List[TimelinePiece[PayloadT]] = []
    cursor = start
    for cover in covers:
        if cursor >= end:
            break
        if cover.end is not None and cover.end <= max(cover.start, cursor):
            continue  # 退化或整体在游标之前的覆盖区间不贡献覆盖
        if cover.start >= end:
            break  # 覆盖区间整体在目标区间之后（后续亦然）
        if cover.start > cursor:
            # 覆盖区间开始前的部分为 gap
            pieces.append(TimelinePiece(start=cursor, end=cover.start, covered=False))
            cursor = cover.start
        overlap_end = end if cover.end is None else min(end, cover.end)
        if overlap_end > cursor:
            pieces.append(
                TimelinePiece(
                    start=cursor, end=overlap_end, covered=True, payload=cover.payload
                )
            )
            cursor = overlap_end
    if cursor < end:
        pieces.append(TimelinePiece(start=cursor, end=end, covered=False))
    return pieces


def ensure_comparable(values: Iterable[datetime], error: Type[Exception]) -> None:
    """混用带时区与不带时区的时间无法比较，统一报错。

    ``error`` 为调用方接口的异常类型（路由层转换为 HTTP 422），
    各接口既有 422 detail 文案保持 ``MIXED_TIMEZONE_MESSAGE`` 不变。
    """
    timestamps = list(values)
    if not timestamps:
        return
    aware = timestamps[0].tzinfo is not None and timestamps[0].utcoffset() is not None
    for value in timestamps[1:]:
        value_aware = value.tzinfo is not None and value.utcoffset() is not None
        if value_aware != aware:
            raise error(MIXED_TIMEZONE_MESSAGE)
