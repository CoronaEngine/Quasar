"""
通用节点工厂

提供构建 LangGraph 节点的工厂函数，简化工作流定义。

核心功能:
1. make_tool_node: 创建调用 StructuredTool 的节点
2. make_transform_node: 创建数据转换节点
3. make_conditional_node: 创建条件分支节点

使用示例:
    from langchain_core.tools import StructuredTool
    from workflow.nodes import make_tool_node

    # 创建调用图像生成工具的节点
    generate_node = make_tool_node(
        tool=image_tool,
        arg_mapper=lambda state: {
            "prompt": state["prompt"],
            "resolution": state["resolution"],
        },
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, TYPE_CHECKING

from .state import WorkflowState

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# 节点函数类型：接收 State，返回部分 State 更新
NodeFunc = Callable[[WorkflowState], Dict[str, Any]]

# 参数映射函数类型：从 State 提取工具参数
ArgMapper = Callable[[WorkflowState], Dict[str, Any]]


def make_tool_node(
    tool: "StructuredTool",
    arg_mapper: ArgMapper,
    *,
    result_key: str = "tool_results",
    error_on_failure: bool = True,
) -> NodeFunc:
    """创建调用 StructuredTool 的节点函数

    Args:
        tool: LangChain StructuredTool 实例
        arg_mapper: 从 State 提取工具参数的函数
        result_key: 存储结果的 State 字段名（默认 "tool_results"）
        error_on_failure: 工具失败时是否设置 error 字段

    Returns:
        节点函数，可直接传给 StateGraph.add_node()

    示例:
        node = make_tool_node(
            tool=generate_image_tool,
            arg_mapper=lambda s: {"prompt": s["prompt"], "resolution": s["resolution"]},
        )
    """

    def node_func(state: WorkflowState) -> Dict[str, Any]:
        # 检查是否已有错误，跳过执行
        if state.get("error"):
            logger.debug(f"Skipping {tool.name} due to previous error")
            return {}

        try:
            # 从 State 提取参数
            args = arg_mapper(state)
            logger.debug(f"Calling tool {tool.name} with args: {args}")

            # 调用工具
            result = tool.func(**args)

            # 将结果追加到列表
            current_results = list(state.get(result_key, []))
            current_results.append(result)

            return {result_key: current_results}

        except Exception as e:
            logger.error(f"Tool {tool.name} failed: {e}")
            if error_on_failure:
                return {"error": f"Tool {tool.name} failed: {str(e)}"}
            return {}

    # 设置函数名便于调试
    node_func.__name__ = f"tool_node_{tool.name}"
    return node_func


def make_transform_node(
    transform: Callable[[WorkflowState], Dict[str, Any]],
    *,
    name: str = "transform",
) -> NodeFunc:
    """创建数据转换节点

    用于在工具调用之间进行数据处理、格式转换等。

    Args:
        transform: 转换函数，接收 State 返回更新字典
        name: 节点名称（用于调试）

    Returns:
        节点函数

    示例:
        # 将中间结果提取到 output_parts
        node = make_transform_node(
            lambda s: {"output_parts": parse_parts(s["tool_results"][-1])},
            name="extract_output",
        )
    """

    def node_func(state: WorkflowState) -> Dict[str, Any]:
        if state.get("error"):
            logger.debug(f"Skipping {name} due to previous error")
            return {}

        try:
            return transform(state)
        except Exception as e:
            logger.error(f"Transform {name} failed: {e}")
            return {"error": f"Transform {name} failed: {str(e)}"}

    node_func.__name__ = f"transform_node_{name}"
    return node_func


def make_prompt_template_node(
    template: str,
    *,
    output_key: str = "prompt",
) -> NodeFunc:
    """创建提示词模板填充节点

    使用 Python format 语法填充模板。

    Args:
        template: 提示词模板（使用 {field} 占位符）
        output_key: 输出字段名（默认覆盖 "prompt"）

    Returns:
        节点函数

    示例:
        node = make_prompt_template_node(
            "为以下产品生成场景图：{prompt}，风格要求：{style}",
        )
    """

    def node_func(state: WorkflowState) -> Dict[str, Any]:
        if state.get("error"):
            return {}

        try:
            # 从 state 和 intermediate 中收集变量
            variables = dict(state)
            variables.update(state.get("intermediate", {}))

            # 填充模板
            filled = template.format(**variables)
            return {output_key: filled}

        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return {"error": f"Missing template variable: {e}"}
        except Exception as e:
            logger.error(f"Template fill failed: {e}")
            return {"error": f"Template fill failed: {str(e)}"}

    node_func.__name__ = "prompt_template_node"
    return node_func


def make_error_check_node() -> NodeFunc:
    """创建错误检查节点

    检查 tool_results 最后一项是否包含错误，
    若有错误则设置 state["error"]。

    Returns:
        节点函数
    """

    def node_func(state: WorkflowState) -> Dict[str, Any]:
        if state.get("error"):
            return {}

        tool_results = state.get("tool_results", [])
        if not tool_results:
            return {}

        import json

        try:
            last_result = tool_results[-1]
            data = (
                json.loads(last_result)
                if isinstance(last_result, str)
                else last_result
            )

            if data.get("error_code", 0) != 0:
                error_msg = data.get("status_info", "Unknown error")
                return {"error": error_msg}

        except (json.JSONDecodeError, TypeError):
            pass

        return {}

    node_func.__name__ = "error_check_node"
    return node_func


__all__ = [
    "NodeFunc",
    "ArgMapper",
    "make_tool_node",
    "make_transform_node",
    "make_prompt_template_node",
    "make_error_check_node",
]
