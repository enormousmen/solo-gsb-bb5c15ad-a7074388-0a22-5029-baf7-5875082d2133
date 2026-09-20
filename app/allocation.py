"""封签库存分配规划。

处理流程：
1. 复用批量校验（``validate_batch``）逐链校验并建立全批次封签占用表，
   收集所有有效（``status=ok``）seal/reseal 事件占用的 ``seal_id``
   （跨链冲突不改变事件有效性，被标记冲突的占用同样计入；违规事件中的
   编号不构成占用）；
2. 库存中编号已被占用的封签直接排除（``excluded_seal_ids``），不参与分配；
3. 对每条待用记录与每个可用库存封签，若保管人在允许名单内且 ``use_at``
   严格早于 ``expires_at``（恰好在过期时刻使用视为已过期），形成一条候选
   边，费用为 ``expires_at - use_at`` 的秒数；
4. 在候选边上求一对一全局最优分配（按待用记录输入顺序的状态压缩动态
   规划，绝非逐条贪心），目标依次为：
     a. 最大化满足的记录数；
     b. 满足数相同时，最小化总费用（Σ expires_at - use_at 秒）；
     c. 仍并列时确定唯一方案：把方案表示为按待用记录输入顺序的分配向量
        （未分配视为排在任何封签之后），取向量字典序最小者——输入靠前的
        记录优先被满足，同一记录并列时优先选择 ``seal_id`` 字典序更小的
        封签；
5. 未分配记录给出原因：无任何候选边为 ``no_candidate``，有候选边但在
   全局最优方案中落选为 ``outcompeted``。

事件、待用记录与库存封签的时间不得混用带时区与不带时区的时间（否则 422）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .batch import validate_batch
from .models import (
    AllocationPlanRequest,
    AllocationPlanResponse,
    AllocationRecordResult,
    AllocationSummary,
    BatchValidateRequest,
    BatchValidateResponse,
    EventType,
)


class AllocationPlanError(ValueError):
    """分配规划入参的语义错误（在路由层转换为 HTTP 422）。"""


_SEAL_EVENTS = (EventType.SEAL, EventType.RESEAL)

# 动态规划状态：(-满足记录数, 总费用秒数, 分配向量)。
# 分配向量按待用记录输入顺序记录每条记录分到的库存顺序下标，未分配记为
# 库存总数（大于任何合法下标）；三者均按“越小越优”逐层比较，一次遍历即得
# 全局最优的唯一方案。
_DpKey = Tuple[int, float, Tuple[int, ...]]


def _ensure_comparable(request: AllocationPlanRequest) -> None:
    """混用带时区与不带时区的时间无法比较，统一报 422。"""
    timestamps = []
    for chain in request.chains:
        timestamps.extend(event.timestamp for event in chain.events)
    timestamps.extend(record.use_at for record in request.pending)
    timestamps.extend(seal.expires_at for seal in request.inventory)
    if not timestamps:
        return
    aware = timestamps[0].tzinfo is not None and timestamps[0].utcoffset() is not None
    for value in timestamps[1:]:
        value_aware = value.tzinfo is not None and value.utcoffset() is not None
        if value_aware != aware:
            raise AllocationPlanError(
                "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"
            )


def _occupied_seal_ids(
    request: AllocationPlanRequest, batch_result: BatchValidateResponse
) -> Set[str]:
    """收集各链有效（status=ok）seal/reseal 事件占用的封签编号。"""
    occupied: Set[str] = set()
    for chain, result in zip(request.chains, batch_result.chains):
        for event, event_result in zip(chain.events, result.events):
            if event_result.status != "ok" or event.type not in _SEAL_EVENTS:
                continue
            if event.seal_id is not None:
                occupied.add(event.seal_id)
    return occupied


def _optimal_allocation(
    candidates: Sequence[Sequence[int]],
    slacks: Sequence[Dict[int, float]],
    n_records: int,
    n_seals: int,
) -> _DpKey:
    """求全局最优分配，返回最优状态键 ``(-满足数, 总费用, 分配向量)``。

    按待用记录输入顺序做状态压缩 DP：``dp[mask]`` 为处理完前 i 条记录、
    已用库存集合为 mask 时的最优键；每条记录可跳过（未分配）或占用一枚
    有候选边且未使用的库存封签。库存顺序下标按 seal_id 字典序编号，
    未分配记为 n_seals（大于任何合法下标），故最终最小键即依次满足
    三级目标的唯一方案。
    """
    unassigned = n_seals
    dp: Dict[int, _DpKey] = {0: (0, 0.0, ())}
    for i in range(n_records):
        nxt: Dict[int, _DpKey] = {}
        for mask, key in dp.items():
            neg_count, total, choices = key
            # 记录 i 不分配
            skip: _DpKey = (neg_count, total, choices + (unassigned,))
            if mask not in nxt or skip < nxt[mask]:
                nxt[mask] = skip
            # 记录 i 分配一枚有候选边且未使用的库存封签
            for j in candidates[i]:
                if mask & (1 << j):
                    continue
                take: _DpKey = (
                    neg_count - 1,
                    total + slacks[i][j],
                    choices + (j,),
                )
                taken = mask | (1 << j)
                if taken not in nxt or take < nxt[taken]:
                    nxt[taken] = take
        dp = nxt
    return min(dp.values())


def plan_allocation(request: AllocationPlanRequest) -> AllocationPlanResponse:
    """复用批量校验排除已占用编号，再求待用记录与库存封签的全局最优分配。"""
    batch_result = validate_batch(BatchValidateRequest(chains=request.chains))
    _ensure_comparable(request)
    occupied = _occupied_seal_ids(request, batch_result)

    # 可用库存：排除已被占用的编号后按 seal_id 字典序排列（顺序下标即字典序）
    available = sorted(
        (seal for seal in request.inventory if seal.seal_id not in occupied),
        key=lambda seal: seal.seal_id,
    )
    excluded = sorted(
        seal.seal_id for seal in request.inventory if seal.seal_id in occupied
    )

    n_records, n_seals = len(request.pending), len(available)
    candidates: List[List[int]] = []
    slacks: List[Dict[int, float]] = []
    for record in request.pending:
        row: List[int] = []
        row_slack: Dict[int, float] = {}
        for j, seal in enumerate(available):
            if record.custodian not in seal.allowed_custodians:
                continue
            if not record.use_at < seal.expires_at:
                continue  # 恰在过期时刻使用视为已过期
            row.append(j)
            row_slack[j] = (seal.expires_at - record.use_at).total_seconds()
        candidates.append(row)
        slacks.append(row_slack)

    _, _, choices = _optimal_allocation(candidates, slacks, n_records, n_seals)

    records: List[AllocationRecordResult] = []
    assigned = no_candidate = outcompeted = 0
    total_slack = 0.0
    for i, record in enumerate(request.pending):
        choice = choices[i]
        if choice < n_seals:
            assigned += 1
            total_slack += slacks[i][choice]
            records.append(
                AllocationRecordResult(
                    index=i,
                    chain_id=record.chain_id,
                    custodian=record.custodian,
                    use_at=record.use_at,
                    status="assigned",
                    seal_id=available[choice].seal_id,
                )
            )
            continue
        if not candidates[i]:
            reason = "no_candidate"
            no_candidate += 1
        else:
            reason = "outcompeted"
            outcompeted += 1
        records.append(
            AllocationRecordResult(
                index=i,
                chain_id=record.chain_id,
                custodian=record.custodian,
                use_at=record.use_at,
                status="unassigned",
                reason=reason,
            )
        )

    return AllocationPlanResponse(
        batch_validation=batch_result,
        excluded_seal_ids=excluded,
        records=records,
        summary=AllocationSummary(
            assigned=assigned,
            unassigned=n_records - assigned,
            no_candidate=no_candidate,
            outcompeted=outcompeted,
            total_slack_seconds=total_slack,
        ),
    )
