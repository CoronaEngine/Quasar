from ..ai_config.ai_types import AIConfig
from typing import Dict, Any
import copy


class ConfigCollector:
    """配置收集器"""

    def __init__(self):
        self._ai_settings = {}
        self._ai_config = AIConfig()
        self._ai_load = {}
        self._ai_prompts = {}
        self._setting_sources = {}

    @staticmethod
    def _is_builtin_setting_source(module_name: str) -> bool:
        return module_name.startswith("ai_modules.") and module_name.endswith(
            ".configs.settings"
        )

    def _should_replace_setting(self, key: str, new_source: str) -> bool:
        current_source = self._setting_sources.get(key)
        if current_source is None:
            return True
        if (
            self._is_builtin_setting_source(new_source)
            and not self._is_builtin_setting_source(current_source)
        ):
            return False
        return True

    def register_setting(self, key: str):
        """装饰器：注册配置函数"""

        def decorator(func):
            source = getattr(func, "__module__", "")
            result = func()
            if not self._should_replace_setting(key, source):
                return func

            self._ai_settings[key] = result
            self._setting_sources[key] = source

            if key in self._ai_load:
                result = self._ai_load[key](self._ai_settings[key])
                setattr(self._ai_config, key, result)

            return func

        return decorator

    def register_prompts(self, key: str):
        """装饰器：注册配置函数"""

        def decorator(func):
            result = func()
            self._ai_prompts[key] = result

        return decorator

    def register_loader(self, key: str):
        def decorator(func):
            self._ai_load[key] = func
            if key in self._ai_settings:
                result = func(self._ai_settings[key])
                setattr(self._ai_config, key, result)

        return decorator

    @property
    def AI_SETTINGS(self) -> Dict[str, Any]:
        return copy.deepcopy(self._ai_settings)

    @property
    def AIConfig(self) -> AIConfig:
        return copy.deepcopy(self._ai_config)
