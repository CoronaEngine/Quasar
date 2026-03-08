"""
工作流状态定义

定义统一的 WorkflowState TypedDict 作为所有工作流的状态基类。
各具体工作流可继承扩展自定义字段。

State 字段说明:
- session_id: 会话 ID，用于关联上下文
- function_id: 功能 ID (10101/10102/10103 等)
- prompt: 用户输入的文本提示词
- images: 输入图片 URL 列表
- resolution: 图片比例 (1:1, 16:9, 3:2 等)
- image_size: 图片分辨率档位 (1K, 2K, 4K)
- tool_results: 工具调用返回的 JSON 字符串列表（按调用顺序）
- output_parts: 最终输出的 part 列表
- intermediate: 中间数据存储（各工作流自定义 key）
- error: 错误信息，非空时表示工作流执行失败
- metadata: 请求携带的元数据（透传）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """工作流状态基类

    使用 total=False 允许字段可选，便于增量更新。
    LangGraph 节点返回部分字段时会自动合并到完整 State。
    """

    # 会话标识
    session_id: str

    # 功能 ID（如 10101, 10102, 10103）
    function_id: int

    # 用户输入
    prompt: str
    images: List[str]  # 图片 URL 列表
    additional_type: Optional[List[str]]  # 额外的输入类型列表
    bounding_box: Optional[List[List[Dict[str, Any]]]]  # 二维数组：bounding_box[i] 对应 images[i] 的 box 列表

    # 生成参数
    resolution: str  # 图片比例
    image_size: str  # 图片分辨率档位

    # 工具调用结果（JSON 字符串列表，按调用顺序存储）
    tool_results: List[str]

    # 中间输出（图像列表，用于工作流节点间传递）
    output_images: List[Dict[str, Any]]

    # 多场景工作流专用字段
    is_multimodal: bool  # 是否包含多模态输入（图片）
    extracted_elements: List[Dict[str, str]]  # analyzer 提取的设计元素
    approved_elements: List[Dict[str, str]]  # 人审通过的设计元素
    generated_images: Dict[str, str]  # 生成的图片 {物品名: URL}
    layout_text: str  # 排版文案

    # 模型检索/生成工作流字段
    model_results: List[Dict[str, Any]]  # 每个物品的检索/生成结果

    # 最终输出
    output_parts: List[Dict[str, Any]]
    output_llm_content: List[Dict[str, Any]]

    # 中间数据（各工作流自定义 key-value）
    intermediate: Dict[str, Any]

    # 错误信息（非空表示失败，后续节点应检查并跳过）
    error: Optional[str]

    # 元数据透传
    metadata: Dict[str, Any]


def create_initial_state(
    *,
    session_id: str,
    function_id: int,
    prompt: str = "",
    images: List[str] | None = None,
    additional_type: List[str] | None = None,
    bounding_box: List[List[Dict[str, Any]]] | None = None,
    resolution: str = "1:1",
    image_size: str = "2K",
    metadata: Dict[str, Any] | None = None,
) -> WorkflowState:
    """创建初始状态

    Args:
        session_id: 会话 ID
        function_id: 功能 ID
        prompt: 文本提示词
        images: 图片 URL 列表
        additional_type: 额外的输入类型列表
        resolution: 图片比例
        image_size: 图片分辨率档位
        metadata: 元数据

    Returns:
        初始化的 WorkflowState
    """
    return WorkflowState(
        session_id=session_id,
        function_id=function_id,
        prompt=prompt,
        images=images or [],
        additional_type=additional_type or [],
        bounding_box=bounding_box or [],
        resolution=resolution,
        image_size=image_size,
        tool_results=[],
        output_images=[],
        output_parts=[],
        intermediate={},
        error=None,
        metadata=metadata or {},
    )


__all__ = ["WorkflowState", "create_initial_state"]
