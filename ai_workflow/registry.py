"""
工作流注册表

管理 function_id 到 CompiledGraph 的映射，支持：
1. 手动注册单个工作流
2. 自动发现 flows/ 目录下的工作流模块

自动发现规则:
- 扫描 flows/ 目录下所有 .py 文件（排除 __init__.py）
- 查找模块中的 WORKFLOWS 变量（Dict[int, CompiledStateGraph]）
- 将所有工作流注册到全局表

使用示例:
    # 手动注册
    registry = get_workflow_registry()
    registry.register(10101, my_compiled_graph)

    # 自动发现
    registry.discover()

    # 获取工作流
    graph = registry.get(10101)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

# 确保 ai_workflow 模块在 Python 路径中
_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 确保 CabbageEditor 目录在 Python 路径中（用于导入 config 等顶级模块）
_cabbage_editor_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _cabbage_editor_dir not in sys.path:
    sys.path.insert(0, _cabbage_editor_dir)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# 单例缓存
_REGISTRY_INSTANCE: Optional["WorkflowRegistry"] = None
_REGISTRY_LOCK = threading.Lock()


class WorkflowRegistry:
    """工作流注册表

    线程安全的单例类，管理 function_id → CompiledStateGraph 映射。
    """

    def __init__(self) -> None:
        self._workflows: Dict[int, "CompiledStateGraph"] = {}
        self._lock = threading.RLock()
        self._discovered = False

    def register(
        self,
        function_id: int,
        graph: "CompiledStateGraph",
        *,
        overwrite: bool = False,
    ) -> None:
        """注册工作流

        Args:
            function_id: 功能 ID (如 10101)
            graph: 已编译的 LangGraph StateGraph
            overwrite: 是否覆盖已存在的注册

        Raises:
            ValueError: 当 function_id 已存在且 overwrite=False
        """
        with self._lock:
            if function_id in self._workflows and not overwrite:
                raise ValueError(
                    f"Workflow for function_id {function_id} already registered. "
                    f"Use overwrite=True to replace."
                )
            self._workflows[function_id] = graph
            logger.debug(f"Registered workflow for function_id={function_id}")

    def get(self, function_id: int) -> Optional["CompiledStateGraph"]:
        """获取工作流

        Args:
            function_id: 功能 ID

        Returns:
            对应的 CompiledStateGraph，未注册时返回 None
        """
        with self._lock:
            return self._workflows.get(function_id)

    def has(self, function_id: int) -> bool:
        """检查工作流是否已注册"""
        with self._lock:
            return function_id in self._workflows

    def list_function_ids(self) -> list[int]:
        """列出所有已注册的 function_id"""
        with self._lock:
            return list(self._workflows.keys())

    def discover(self, *, force: bool = False) -> int:
        """自动发现并注册工作流

        执行顺序：
        1. 扫描 flows/ 目录下的内置工作流模块
        2. 尝试加载外部工作流（InnerAgentWorkflow/ai_workflows）

        外部工作流可以覆盖内置工作流（支持私有定制场景）。

        Args:
            force: 是否强制重新发现（默认只发现一次）

        Returns:
            新注册的工作流数量
        """
        with self._lock:
            if self._discovered and not force:
                logger.debug("Workflows already discovered, skipping")
                return 0

            count = 0

            # 1. 扫描内置 flows/ 目录
            count += self._discover_builtin_workflows()

            # 2. 加载外部工作流（可覆盖内置）
            count += self._discover_external_workflows()

            self._discovered = True
            logger.info(
                f"Workflow discovery complete: {count} workflow(s) registered "
                f"(function_ids={self.list_function_ids()})"
            )
            return count

    def _discover_builtin_workflows(self) -> int:
        """发现内置工作流（flows/ 目录）

        Returns:
            新注册的工作流数量
        """
        count = 0
        flows_path = Path(__file__).parent / "flows"

        if not flows_path.exists():
            logger.info(f"Flows directory not found: {flows_path}")
            return 0

        # 扫描 flows 目录下的模块
        package_name = "ai_workflow.flows"

        for module_info in pkgutil.iter_modules([str(flows_path)]):
            if module_info.name.startswith("_"):
                continue

            module_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)

                # 查找 WORKFLOWS 变量
                workflows = getattr(module, "WORKFLOWS", None)
                if workflows is None:
                    logger.debug(f"No WORKFLOWS found in {module_name}")
                    continue

                if not isinstance(workflows, dict):
                    logger.warning(
                        f"WORKFLOWS in {module_name} is not a dict, skipping"
                    )
                    continue

                # 注册所有工作流
                for fid, graph in workflows.items():
                    if not isinstance(fid, int):
                        logger.warning(
                            f"Invalid function_id type in {module_name}: {type(fid)}"
                        )
                        continue

                    try:
                        self.register(fid, graph, overwrite=False)
                        count += 1
                    except ValueError as e:
                        logger.debug(f"Skip duplicate: {e}")

            except Exception as e:
                logger.error(f"Failed to load workflow module {module_name}: {e}")

        if count > 0:
            logger.debug(f"Discovered {count} builtin workflow(s) from flows/")

        return count

    def _discover_external_workflows(self) -> int:
        """发现外部工作流（来自 InnerAgentWorkflow/ai_workflows）

        外部工作流会覆盖同 function_id 的内置工作流。

        Returns:
            新注册的工作流数量
        """
        count = 0

        try:
            # 尝试导入外部工作流模块
            external_module = importlib.import_module("workflows")

            # 查找 load_external_workflows 函数
            load_fn = getattr(external_module, "load_external_workflows", None)
            if load_fn is None:
                logger.debug(
                    "No load_external_workflows found in ai_workflows"
                )
                return 0

            # 加载工作流
            workflows = load_fn()

            if not isinstance(workflows, dict):
                logger.warning("load_external_workflows did not return a dict")
                return 0

            # 注册工作流（覆盖已存在的）
            for fid, graph in workflows.items():
                if not isinstance(fid, int):
                    logger.warning(f"Invalid function_id type from external: {type(fid)}")
                    continue

                try:
                    # 外部工作流可以覆盖内置工作流
                    overwrite = self.has(fid)
                    self.register(fid, graph, overwrite=overwrite)

                    if overwrite:
                        logger.debug(f"External workflow overrides builtin: {fid}")

                    count += 1
                except Exception as e:
                    logger.error(f"Failed to register external workflow {fid}: {e}")

            if count > 0:
                logger.debug(f"Loaded {count} external workflow(s)")

        except ImportError:
            logger.debug("ai_workflows not available")
        except Exception as e:
            logger.error(f"Failed to load external workflows: {e}")

        return count

    def clear(self) -> None:
        """清空注册表（主要用于测试）"""
        with self._lock:
            self._workflows.clear()
            self._discovered = False


def get_workflow_registry() -> WorkflowRegistry:
    """获取工作流注册表单例"""
    global _REGISTRY_INSTANCE

    if _REGISTRY_INSTANCE is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_INSTANCE is None:
                _REGISTRY_INSTANCE = WorkflowRegistry()

    return _REGISTRY_INSTANCE


__all__ = ["WorkflowRegistry", "get_workflow_registry"]
