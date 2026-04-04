from __future__ import annotations

import ast
import importlib
import logging
import pkgutil
import threading

from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_COMMAND_REGISTRY_INSTANCE: Optional["WorkflowCommandRegistry"] = None
_COMMAND_REGISTRY_LOCK = threading.Lock()


class WorkflowCommandRegistry:
    """工作流命令注册表，管理命令到 function_id 的映射。"""

    def __init__(self) -> None:
        self._commands: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._discovered = False

    @staticmethod
    def _normalize_command(command: str) -> str:
        normalized = command.strip().lower()
        if normalized and not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def register(self, command: str, function_id: int, *, overwrite: bool = False) -> None:
        normalized = self._normalize_command(command)
        if not normalized:
            raise ValueError("command 不能为空")

        with self._lock:
            existing = self._commands.get(normalized)
            if existing is not None and existing != function_id and not overwrite:
                raise ValueError(
                    f"Command {normalized} already registered for function_id={existing}."
                )
            self._commands[normalized] = function_id
            logger.debug(
                "Registered workflow command %s -> %s",
                normalized,
                function_id,
            )

    def resolve(self, command: str) -> Optional[int]:
        normalized = self._normalize_command(command)
        if not normalized:
            return None
        with self._lock:
            return self._commands.get(normalized)

    def list_commands(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._commands)

    @staticmethod
    def _extract_workflow_commands_from_source(module_path: Path) -> Dict[str, int]:
        """从源码静态提取 WORKFLOW_COMMANDS，避免导入时依赖链失败。"""
        try:
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module_path))
        except Exception as exc:
            logger.debug("Failed to parse workflow source %s: %s", module_path, exc)
            return {}

        constants: Dict[str, int] = {}

        def _extract_name(node: ast.AST) -> Optional[str]:
            if isinstance(node, ast.Name):
                return node.id
            return None

        def _extract_int(node: ast.AST) -> Optional[int]:
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return int(node.value)
            if isinstance(node, ast.Name):
                return constants.get(node.id)
            return None

        for stmt in tree.body:
            target_name: Optional[str] = None
            value_node: Optional[ast.AST] = None

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target_name = _extract_name(stmt.targets[0])
                value_node = stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target_name = _extract_name(stmt.target)
                value_node = stmt.value

            if not target_name or value_node is None:
                continue

            value = _extract_int(value_node)
            if value is not None:
                constants[target_name] = value

        for stmt in tree.body:
            target_name: Optional[str] = None
            value_node: Optional[ast.AST] = None

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target_name = _extract_name(stmt.targets[0])
                value_node = stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target_name = _extract_name(stmt.target)
                value_node = stmt.value

            if target_name != "WORKFLOW_COMMANDS" or not isinstance(value_node, ast.Dict):
                continue

            extracted: Dict[str, int] = {}
            for key_node, value_node in zip(value_node.keys, value_node.values):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                resolved_function_id = _extract_int(value_node)
                if resolved_function_id is None:
                    continue
                extracted[str(key_node.value)] = resolved_function_id

            return extracted

        return {}

    def discover(self, *, force: bool = False) -> int:
        """扫描 flows 模块中的 WORKFLOW_COMMANDS 并注册。"""
        with self._lock:
            if self._discovered and not force:
                return 0

            count = 0
            flows_path = Path(__file__).parent / "flows"
            if not flows_path.exists():
                self._discovered = True
                return 0

            package_name = "ai_workflow.flows"
            for module_info in pkgutil.iter_modules([str(flows_path)]):
                if module_info.name.startswith("_"):
                    continue

                module_name = f"{package_name}.{module_info.name}"
                module_path = flows_path / f"{module_info.name}.py"
                try:
                    module = importlib.import_module(module_name)
                    workflow_commands = getattr(module, "WORKFLOW_COMMANDS", None)
                except Exception as exc:
                    logger.warning(
                        "Failed to load workflow module %s, fallback to source parse: %s",
                        module_name,
                        exc,
                    )
                    workflow_commands = self._extract_workflow_commands_from_source(module_path)

                if workflow_commands is None:
                    continue
                if not isinstance(workflow_commands, dict):
                    logger.warning("WORKFLOW_COMMANDS in %s is not a dict", module_name)
                    continue

                for command, function_id in workflow_commands.items():
                    if not isinstance(command, str) or not isinstance(function_id, int):
                        logger.warning(
                            "Invalid command mapping in %s: %r -> %r",
                            module_name,
                            command,
                            function_id,
                        )
                        continue
                    try:
                        self.register(command, function_id, overwrite=False)
                        count += 1
                    except ValueError as exc:
                        logger.warning("Skip duplicate workflow command in %s: %s", module_name, exc)

            self._discovered = True
            logger.info("Workflow command discovery complete: %s command(s)", len(self._commands))
            return count


def get_workflow_command_registry() -> WorkflowCommandRegistry:
    global _COMMAND_REGISTRY_INSTANCE

    if _COMMAND_REGISTRY_INSTANCE is None:
        with _COMMAND_REGISTRY_LOCK:
            if _COMMAND_REGISTRY_INSTANCE is None:
                _COMMAND_REGISTRY_INSTANCE = WorkflowCommandRegistry()

    return _COMMAND_REGISTRY_INSTANCE


__all__ = ["WorkflowCommandRegistry", "get_workflow_command_registry"]
