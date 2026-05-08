"""
目标检测服务模块

提供独立的目标检测接口，使用 VLM 进行图像目标检测，
返回主体对象的边界框（相对坐标系，原点左下角）和描述。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ...ai_config.ai_config import get_ai_config
from ...ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    parse_tool_response,
)
from ...ai_tools.concurrency import session_concurrency
from ...ai_service.entrance import register_entrance
from ...ai_tools.context import set_current_session, reset_current_session
from ...ai_tools.helpers import request_time_diff
from ...ai_tools.request_parser import (
    extract_prompt_from_llm_content,
    extract_images_from_request,
)
from ...ai_tools.session_tracking import init_session, update_session_state

logger = logging.getLogger(__name__)


@register_entrance(handler_name="handle_detection")
def handle_detection(payload: Any) -> str:
    """目标检测接口，返回标准三层结构。

    输入格式 (llm_content):
    - part[].content_type: "image" - 待检测的图片
    - part[].content_type: "text" - 可选的目标描述（如"人物"、"汽车"）

    输出格式 (llm_content):
    - part[].content_type: "detection"
    - part[].content_text: 汇总描述
    - part[].content_url: 检测的源图片 URL
    - part[].parameter.bounding_box: [{postion: [...], describe: "...", label: "..."}] 包围盒数组
      - postion: [x_min, y_min, x_max, y_max]，归一化坐标 0~1，原点左上角
      - describe: 对象详细描述
      - label: 对象类别名称
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="detection",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_detection_inner(request_data, session_id, metadata, cfg)


def _handle_detection_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """目标检测内部实现（在并发控制内执行）"""
    # 设置会话上下文（用于工具获取当前会话）
    token = set_current_session(session_id)

    try:
        # 初始化会话追踪（确保账户使用记录可以正常工作）
        init_session(
            session_id=session_id,
            input_type="detection",
            parameters={
                "metadata": metadata,
            },
        )
        update_session_state(session_id, "running")

        logger.debug(f"收到目标检测请求: {request_data}")

        # 提取图片 URL
        image_urls = extract_images_from_request(request_data)
        image_url = image_urls[0] if image_urls else ""
        if not image_url:
            raise ValueError("缺少待检测的图片，请在 part 中提供 content_type 为 'image' 的图片 URL")

        # 提取目标描述（可选）
        target_description = extract_prompt_from_llm_content(request_data)

        # 加载检测工具（从外部模块）
        from tools.detection_tools import (
            load_detection_tools,
        )

        tools = load_detection_tools(cfg)
        if not tools:
            raise RuntimeError("目标检测功能未启用或配置不完整")

        detection_tool = tools[0]
        result_json = detection_tool.invoke(
            {
                "image_url": image_url,
                "target_description": target_description,
            },
            config={"session_id": session_id},
        )
        logger.debug(f"detection_tool 返回: {result_json}")

        # 解析 Tool 返回的 Envelope JSON
        tool_envelope = parse_tool_response(result_json)
        logger.debug(f"解析 tool_envelope: {tool_envelope}")

        # 检查错误
        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            logger.error(f"目标检测失败: {error_msg}")
            raise RuntimeError(f"目标检测失败: {error_msg}")

        # 提取 llm_content
        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            logger.error("目标检测未返回有效内容")
            raise RuntimeError("目标检测未返回有效内容")

        # 提取并清洗 parts
        original_parts = llm_content[0].get("part", [])
        cleaned_parts = _clean_detection_parts(original_parts)
        logger.debug(f"清洗后的 parts: {cleaned_parts}")

        # 更新会话状态为完成
        update_session_state(session_id, "completed")

        return build_success_response(
            interface_type="detection",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )

    except Exception as exc:
        logger.error(f"目标检测异常: {exc}")
        # 更新会话状态为失败
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type="detection",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
    finally:
        # 重置会话上下文
        reset_current_session(token)


def _clean_detection_parts(original_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """清洗检测结果 parts，确保符合 API 标准格式。

    输出格式：
    - content_type: "detection"
    - content_text: 汇总描述
    - content_url: 检测的图片 URL（可选）
    - parameter.bounding_box: [{postion: [...], describe: "...", label: "..."}] 包围盒数组
    """
    cleaned_parts = []

    for part in original_parts:
        if part.get("content_type") != "detection":
            continue

        cleaned_part: Dict[str, Any] = {
            "content_type": "detection",
            "content_text": part.get("content_text", ""),
        }

        # 保留 content_url（检测的源图片）
        if "content_url" in part and part["content_url"]:
            cleaned_part["content_url"] = part["content_url"]

        # 提取并验证 parameter
        original_param = part.get("parameter", {})
        if isinstance(original_param, dict):
            cleaned_param: Dict[str, Any] = {}

            # bounding_box: 应为对象数组 [{postion: [...], describe: "...", label: "..."}]
            if "bounding_box" in original_param:
                bbox = original_param["bounding_box"]
                if isinstance(bbox, list):
                    cleaned_boxes = []
                    for item in bbox:
                        if isinstance(item, dict):
                            # 新格式：{postion: [...], describe: "...", label: "..."}
                            box_item = {}
                            if "postion" in item:
                                pos = item["postion"]
                                if isinstance(pos, list) and len(pos) == 4:
                                    try:
                                        box_item["postion"] = [
                                            round(float(v), 4) for v in pos
                                        ]
                                    except (TypeError, ValueError):
                                        continue
                            if "describe" in item:
                                box_item["describe"] = str(item["describe"])
                            if "label" in item:
                                box_item["label"] = str(item["label"])
                            if box_item:
                                cleaned_boxes.append(box_item)
                        elif isinstance(item, (int, float)):
                            # 旧格式兼容：[x_min, y_min, x_max, y_max] 简单数组
                            # 整个 bbox 是一个简单坐标数组，转换为新格式
                            if len(bbox) == 4:
                                try:
                                    cleaned_boxes = [{
                                        "postion": [round(float(v), 4) for v in bbox],
                                        "describe": cleaned_part.get("content_text", ""),
                                    }]
                                except (TypeError, ValueError):
                                    pass
                            break
                    if cleaned_boxes:
                        cleaned_param["bounding_box"] = cleaned_boxes

            if cleaned_param:
                cleaned_part["parameter"] = cleaned_param

        cleaned_parts.append(cleaned_part)

    return cleaned_parts


__all__ = ["handle_detection"]
