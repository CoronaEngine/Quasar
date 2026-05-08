"""
会话追踪子模块

提供会话进度追踪和账户使用记录功能。
"""

from .models import (
    StepRetryInfo,
    StepInfo,
    SessionProgress,
    AccountUsageRecord,
    DeadlineInfo,
    SessionCache,
)
from .cache import (
    SessionCacheManager,
    get_session_cache_manager,
)

# 便捷函数（直接调用单例管理器的方法）
_manager = None


def _get_manager() -> SessionCacheManager:
    global _manager
    if _manager is None:
        _manager = get_session_cache_manager()
    return _manager


# 写入接口
def init_session(
    session_id: str,
    input_type: str,
    parameters: dict,
    workflow_state: dict | None = None,
) -> None:
    """工作流启动时初始化会话

    Args:
        session_id: 会话ID
        input_type: 输入类型
        parameters: 输入参数字典
        workflow_state: 工作流初始状态（可选）
    """
    _get_manager().init_session(session_id, input_type, parameters, workflow_state)


def update_session_state(session_id: str, state: str) -> None:
    """工作流更新状态"""
    _get_manager().update_state(session_id, state)


def update_session_progress(
    session_id: str,
    current_step: int,
    total_steps: int,
    step_name: str,
    message: str,
    progress_percent: float | None = None,
) -> None:
    """工作流更新进度"""
    _get_manager().update_progress(
        session_id, current_step, total_steps, step_name, message, progress_percent
    )


def record_step_start(
    session_id: str,
    step_name: str,
    step_number: int,
    attempt: int = 1,
    max_attempts: int = 3,
) -> None:
    """工作流记录步骤开始"""
    _get_manager().record_step_start(
        session_id, step_name, step_number, attempt, max_attempts
    )


def record_step_retry(
    session_id: str, step_name: str, step_number: int, error: str, next_attempt: int
) -> None:
    """工作流记录重试"""
    _get_manager().record_step_retry(
        session_id, step_name, step_number, error, next_attempt
    )


def record_step_complete(
    session_id: str,
    step_name: str,
    step_number: int,
    success: bool,
    error: str | None = None,
) -> None:
    """工作流记录步骤完成"""
    _get_manager().record_step_complete(
        session_id, step_name, step_number, success, error
    )


def append_session_output(session_id: str, output_type: str, content: dict) -> None:
    """工作流添加输出结果"""
    _get_manager().append_output(session_id, output_type, content)


def set_session_error(session_id: str, error: str) -> None:
    """工作流记录错误"""
    _get_manager().set_error(session_id, error)


def record_account_usage_to_session(
    session_id: str,
    account_id: str,
    account_name: str,
    model: str | None,
    price: float,
    latency_ms: float,
    success: bool,
) -> None:
    """记录账户使用到会话缓存"""
    _get_manager().record_account_usage(
        session_id, account_id, account_name, model, price, latency_ms, success
    )


def record_deadline_info(
    session_id: str,
    deadline: float,
    start_time: float,
    stage_timings: list,
    success: bool,
    is_timeout: bool,
) -> None:
    """记录 Deadline 超时管理信息到会话缓存"""
    _get_manager().record_deadline_info(
        session_id, deadline, start_time, stage_timings, success, is_timeout
    )


def get_deadline_info(session_id: str) -> DeadlineInfo | None:
    """获取会话的 Deadline 信息"""
    return _get_manager().get_deadline_info(session_id)


# 读取接口
def get_session_status(session_id: str) -> dict:
    """API查询会话状态"""
    return _get_manager().get_status(session_id)


def get_session_progress(session_id: str) -> dict:
    """API查询进度详情"""
    return _get_manager().get_progress(session_id)


def get_session_input(session_id: str) -> dict:
    """API查询输入参数"""
    return _get_manager().get_input(session_id)


def get_session_output(session_id: str) -> dict:
    """API查询输出结果"""
    return _get_manager().get_output(session_id)


def get_session_snapshot(session_id: str) -> dict:
    """API查询会话完整快照"""
    return _get_manager().get_snapshot(session_id)


def get_session_accounts(session_id: str) -> dict:
    """API查询会话账户使用信息"""
    return _get_manager().get_accounts(session_id)


__all__ = [
    # 数据结构
    "StepRetryInfo",
    "StepInfo",
    "SessionProgress",
    "AccountUsageRecord",
    "DeadlineInfo",
    "SessionCache",
    # 管理器
    "SessionCacheManager",
    "get_session_cache_manager",
    # 写入接口
    "init_session",
    "update_session_state",
    "update_session_progress",
    "record_step_start",
    "record_step_retry",
    "record_step_complete",
    "append_session_output",
    "set_session_error",
    "record_account_usage_to_session",
    "record_deadline_info",
    "get_deadline_info",
    # 读取接口
    "get_session_status",
    "get_session_progress",
    "get_session_input",
    "get_session_output",
    "get_session_snapshot",
    "get_session_accounts",
]
