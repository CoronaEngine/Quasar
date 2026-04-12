from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

BaseDir = "D:\\CodeLib\\New_Corona_Project\\"

SOFA_IMAGE_PATH = (
    f"{BaseDir}media\\Gemini_Generated_Image_4crmdj4crmdj4crm.jpg"
)

LAMP_IMAGE_PATH = (
    f"{BaseDir}media\\Gemini_Generated_Image_c29wvcc29wvcc29w.jpg"
)

APPROVED_ELEMENTS = [
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
]

GENERATED_IMAGES = {
    "现代沙发": SOFA_IMAGE_PATH,
    "艺术落地灯": LAMP_IMAGE_PATH,
}

TEST_CASE_DATA: Dict[str, Dict[str, Any]] = {
    "default": {
        "global_assets": {
            "multi_scene": {
                "approved_elements": APPROVED_ELEMENTS,
                "generated_images": GENERATED_IMAGES,
            }
        },
        "expected_model_results": [
            {
                "item_name": "现代沙发",
                "object_id": "modern_sofa",
                "task_index": 1,
                "source": "retrieval",
                "distance": 0.15,
                "model_path": f"{BaseDir}models\\现代沙发\\现代沙发.glb",
                "input_image_url": SOFA_IMAGE_PATH,
                "image_paths": [SOFA_IMAGE_PATH],
                "register_status": "skipped",
            },
            {
                "item_name": "艺术落地灯",
                "object_id": "art_lamp",
                "task_index": 2,
                "source": "generation",
                "model_path": f"{BaseDir}models\\艺术落地灯\\艺术落地灯.glb",
                "input_image_url": LAMP_IMAGE_PATH,
                "preview_paths": [LAMP_IMAGE_PATH],
                "register_status": "inserted",
            },
        ],
    },
    "input_only": {
        "global_assets": {
            "multi_scene": {
                "approved_elements": APPROVED_ELEMENTS,
                "generated_images": GENERATED_IMAGES,
            }
        },
    },
    "six_view_ready": {
        "global_assets": {
            "multi_scene": {
                "approved_elements": APPROVED_ELEMENTS,
                "generated_images": GENERATED_IMAGES,
            }
        },
        "expected_model_results": [
            {
                "item_name": "现代沙发",
                "object_id": "modern_sofa",
                "task_index": 1,
                "source": "generation",
                "model_path": f"{BaseDir}models\\现代沙发\\现代沙发.glb",
                "input_image_url": SOFA_IMAGE_PATH,
                "mock_visual_review": "PASS",
            },
            {
                "item_name": "艺术落地灯",
                "object_id": "art_lamp",
                "task_index": 2,
                "source": "generation",
                "model_path": f"{BaseDir}models\\艺术落地灯\\艺术落地灯.glb",
                "input_image_url": LAMP_IMAGE_PATH,
            },
        ],
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


# 导出兼容接口
TEST_CASES = TEST_CASE_DATA
