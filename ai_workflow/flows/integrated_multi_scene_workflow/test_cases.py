from __future__ import annotations

import logging
from typing import Any, Dict

from .constants import FALLBACK_ELEMENTS

logger = logging.getLogger(__name__)

TEST_CASE_DATA: Dict[str, Dict[str, Any]] = {
    "default": {
        "extracted_elements": FALLBACK_ELEMENTS,
        "approved_elements": FALLBACK_ELEMENTS,
        "generated_images": {
            "现代沙发": "file://test_sofa.jpg",
            "艺术落地灯": "file://test_lamp.jpg",
            "装饰画": "file://test_art.jpg",
        },
        "layout_text": (
            "## 现代客厅设计方案\n"
            "1. 现代沙发\n"
            "   放置于客厅中央，搭配浅色地毯与茶几。\n"
            "2. 艺术落地灯\n"
            "   置于沙发侧旁，提供柔和氛围照明。\n"
            "3. 装饰画\n"
            "   悬挂于沙发上方墙面，作为空间视觉焦点。"
        ),
    },
    "input_only": {
        "input_prompt": "请为我设计一个现代轻奢风格的客厅方案，包含主要家具、照明和墙面装饰。",
    },
    "partial_elements": {
        "extracted_elements": [
            {
                "item_name": "北欧风餐桌",
                "image_prompt": (
                    "A Scandinavian dining table, light wood, minimalist "
                    "design, isolated on white background"
                ),
                "layout_desc": "放置于餐厅中央，与开放厨房相邻。",
            }
        ],
        "approved_elements": None,
        "generated_images": {},
    },
}


def get_test_case(test_case_key: str) -> Dict[str, Any]:
    """获取指定测试样例的 state 覆盖数据。"""
    case_data = TEST_CASE_DATA.get(test_case_key or "default", {})
    logger.info(
        "[MultiScene][test_case] Loaded test case: %s, fields=%s",
        test_case_key or "default",
        list(case_data.keys()),
    )
    return dict(case_data)


# 导出兼容接口
TEST_CASES = TEST_CASE_DATA
