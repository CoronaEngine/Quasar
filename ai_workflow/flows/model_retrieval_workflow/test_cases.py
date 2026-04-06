from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

TEST_CASE_DATA: Dict[str, Dict[str, Any]] = {
    "default": {
        "global_assets": {
            "multi_scene": {
                "approved_elements": [
                    {
                        "item_name": "现代沙发",
                        "image_prompt": "A modern minimalist sofa...",
                        "layout_desc": "放置于客厅中央",
                    },
                    {
                        "item_name": "艺术落地灯",
                        "image_prompt": "An artistic floor lamp...",
                        "layout_desc": "置于沙发侧旁",
                    },
                ],
                "generated_images": {
                    "现代沙发": "file://test_sofa.jpg",
                    "艺术落地灯": "file://test_lamp.jpg",
                },
            }
        },
        "expected_model_results": [
            {
                "item_name": "现代沙发",
                "object_id": "modern_sofa",
                "source": "retrieval",
                "distance": 0.15,
            },
            {
                "item_name": "艺术落地灯",
                "object_id": "art_lamp",
                "source": "generation",
                "model_path": "/models/art_lamp/base.glb",
            },
        ],
    },
    "input_only": {
        "global_assets": {
            "multi_scene": {
                "approved_elements": [
                    {
                        "item_name": "现代沙发",
                        "image_prompt": (
                            "A modern minimalist sofa with clean lines, premium "
                            "fabric, isolated on white background"
                        ),
                        "layout_desc": "放置于客厅中央，形成主要会客区。",
                    },
                    {
                        "item_name": "艺术落地灯",
                        "image_prompt": (
                            "An artistic floor lamp, contemporary design, warm "
                            "ambient style, isolated on white background"
                        ),
                        "layout_desc": "置于沙发侧后方，提供辅助氛围照明。",
                    },
                ],
                "generated_images": {
                    "现代沙发": (
                        "D:\\CodeLib\\storage_root\\media_storage\\"
                        "resource_4dafc270e3e44c7ea514215b406b80ab_"
                        "c4efd90a-31a1-11f1-b3e9-68ecc582fadb.png"
                    ),
                    "艺术落地灯": (
                        "D:\\CodeLib\\storage_root\\media_storage\\"
                        "resource_4dafc270e3e44c7ea514215b406b80ab_"
                        "c032ee4b-31a1-11f1-bbb2-68ecc582fadb.png"
                    ),
                },
            }
        },
    },
}


def get_test_case(test_case_key: str) -> Dict[str, Any]:
    """获取指定测试样例的完整 state 覆盖数据。"""
    case_data = TEST_CASE_DATA.get(test_case_key or "default", {})
    logger.info(
        "[ModelRetrieval][test_case] Loaded test case: %s, fields=%s",
        test_case_key or "default",
        list(case_data.keys()),
    )
    return dict(case_data)
