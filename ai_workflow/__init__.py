"""
Workflow 模块 - 基于 LangGraph 的预定义工作流编排

此模块独立于 Agent 层，直接调用 Tools 层的 StructuredTool，
通过 StateGraph 编排预定义的节点执行顺序。

架构位置:
    Service Layer → Workflow Layer → Tools Layer
                 ↘ Agent Layer ↗

核心组件:
- state: 工作流状态定义
- registry: 工作流注册表（自动发现）
- executor: 工作流执行器
- adapter: 输入输出适配器
- nodes: 通用节点工厂
- flows/: 具体工作流实现

使用方式:
    from workflow import run_workflow

    result = run_workflow(function_id=10101, request_data=payload)
    if result is None:
        # function_id 未注册，fallback 到原有路径
        ...
"""

from .executor import run_workflow
from .registry import (
    WorkflowRegistry,
    get_workflow_registry,
)
from .state import BaseWorkflowState, WorkflowState
from .bridge import (
    RequestContext,
    extract_text,
    parse_command,
    normalize_int_function_id,
    resolve_workflow_command,
    parse_request_context,
    inject_function_id_and_prompt,
)

__all__ = [
    "run_workflow",
    "WorkflowRegistry",
    "get_workflow_registry",
    "BaseWorkflowState",
    "WorkflowState",
    "RequestContext",
    "extract_text",
    "parse_command",
    "normalize_int_function_id",
    "resolve_workflow_command",
    "parse_request_context",
    "inject_function_id_and_prompt",
]
