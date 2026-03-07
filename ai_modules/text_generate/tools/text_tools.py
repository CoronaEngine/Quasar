"""
文案与场景生成工具 - 整合多风格规划、空间拆解与画图提示词提取
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from ai_config.ai_config import AIConfig
from ai_models.base_pool import get_chat_model

from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)


from ai_modules.text_generate.configs.prompts import (
    PRODUCT_TEXT_PROMPTS, 
    MARKETING_TEXT_PROMPTS, 
    CREATIVE_TEXT_PROMPTS,
    PLATFORM_TIPS,
)

# ==========================================
# 1. 定义参数模式 (Schemas)
# ==========================================

class ProductTextInput(BaseModel):
    
    instruction: str = Field(..., description=PRODUCT_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(default="专业", description=PRODUCT_TEXT_PROMPTS.fields["style"])
    length: str = Field(default="中等", description=PRODUCT_TEXT_PROMPTS.fields["length"])
    

class MarketingTextInput(BaseModel):
    
    instruction: str = Field(..., description=MARKETING_TEXT_PROMPTS.fields["instruction"])
    platform: str = Field(default="通用", description=MARKETING_TEXT_PROMPTS.fields["platform"])
    tone: str = Field(default="激励", description=MARKETING_TEXT_PROMPTS.fields["tone"])

class CreativeTextInput(BaseModel):
    
    instruction: str = Field(..., description=CREATIVE_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(default="现代", description=CREATIVE_TEXT_PROMPTS.fields["style"])
    length: str = Field(default="中等", description=CREATIVE_TEXT_PROMPTS.fields["length"])
    

class ScenePlanInput(BaseModel):
    """场景规划与拆解（终极整合版）输入参数"""
    scene_type: str = Field(..., description="场景类型，例如：卧室、电竞房、客厅等")
    style: str = Field(default="现代", description="首选的设计风格")
    detail_level: str = Field(default="中等", description="细节丰富程度")
    constraints: Optional[str] = Field(default=None, description="特定的约束条件，如尺寸、颜色等")
    style_count: int = Field(default=1, description="需要生成的设计方案/风格数量，默认1种")
    
    # 兼容老字段
    views: List[str] = Field(default_factory=lambda: ["overall"])
    image_size: str = Field(default="2K")
    resolution: str = Field(default="1:1")


# ==========================================
# 2. 工具加载与执行逻辑
# ==========================================

def load_text_tools(config: AIConfig) -> List[StructuredTool]:
    
    llm = get_chat_model(category="text", temperature=0.8, request_timeout=60.0)

    def _extract_json_block(text: str) -> Tuple[str, Optional[dict]]:
        
        if not text:
            return "", None
        
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if m:
            json_str = m.group(1).strip()
            readable = (text[: m.start()] + text[m.end() :]).strip()
            try:
                return readable, json.loads(json_str)
            except Exception:
                return text.strip(), None
            
        m2 = re.search(r"(\{[\s\S]*\})\s*$", text.strip())
        if m2:
            try:
                obj = json.loads(m2.group(1))
                readable = text[: m2.start()].strip()
                return readable, obj
            except Exception:
                pass
            
        return text.strip(), None

    def _process_generation(system_prompt: str, user_prompt: str, text_type: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            response = llm.invoke(messages)
            part = build_part(
                content_type="text",
                content_text=response.content,
                parameter={"additional_type": [text_type]},
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="text")
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    def _generate_product_text(instruction: str, style: str = "专业", length: str = "中等") -> str:
        length_map = {"简短": "50-80字", "中等": "150-200字", "详细": "300-500字"}
        prompt = PRODUCT_TEXT_PROMPTS.user_prompt.format(
            style=style, length_hint=length_map.get(length, "150-200字"), instruction=instruction
        )
        return _process_generation(PRODUCT_TEXT_PROMPTS.system_prompt, prompt, "product_text")

    def _generate_marketing_text(instruction: str, platform: str = "通用", tone: str = "激励") -> str:
        prompt = MARKETING_TEXT_PROMPTS.user_prompt.format(
            tone=tone, instruction=instruction, platform=platform, platform_tip=PLATFORM_TIPS.get(platform, "")
        )
        return _process_generation(MARKETING_TEXT_PROMPTS.system_prompt, prompt, "marketing_text")
    
    def _generate_creative_text(instruction: str, style: str = "现代", length: str = "中等") -> str:
        length_map = {"简短": "100字以内", "中等": "300-500字", "长篇": "800-1000字"}
        prompt = CREATIVE_TEXT_PROMPTS.user_prompt.format(
            style=style, instruction=instruction, length_hint=length_map.get(length, "300-500字")
        )
        return _process_generation(CREATIVE_TEXT_PROMPTS.system_prompt, prompt, "creative_text")

    # 核心整合工具：替代原有的 plan 和 breakdown
    def _generate_scene_plan(
        scene_type: str, style: str = "现代", detail_level: str = "中等",
        constraints: Optional[str] = None, views: Optional[List[str]] = None,
        image_size: str = "2K", resolution: str = "1:1", style_count: int = 1
    ) -> str:
        constraints_str = constraints or "无"
        
        system_prompt = f"""你是一个顶级的室内设计师和场景规划师。
请根据用户的场景类型【{scene_type}】，提供 {style_count} 种不同的设计方案。

你必须同时输出“可读文本”和“JSON数据”。

=== A) 可读文本 ===
对于这 {style_count} 种风格中的每一种，请严格按如下结构输出：

### 风格 [序号]: [风格名称]
**1. 场景单个物品列表清单**：
(详细列出该场景/房间内所有核心的单体物品)

**2. 单个物体布局的文字描述**：
(详细描述上述清单中每一个物品在空间中的具体摆放位置、朝向以及与其他物品的关系)

**3. 单体物品画图提示词**：
(为上述清单中的核心物品，分别写出一句纯中文的AI绘画Prompt)
[IMAGE_PROMPT] 一个单独的[物品详细描述]，[当前风格]风格，纯白背景，影棚级别光照，3D渲染，极高画质
[IMAGE_PROMPT] 一个单独的[物品详细描述]，[当前风格]风格，纯净白底，专业产品摄影，高清细节，居中展示

=== B) 结构化 JSON（必须放在 ```json 代码块内```） ===
{{
  "scene_type": "{scene_type}",
  "detail_level": "{detail_level}",
  "plans": [
    {{
      "style": "风格名称",
      "objects": [
        {{"name": "物品名称", "layout": "位置布局描述"}}
      ]
    }}
  ]
}}

【硬性规则】
- 必须严格输出 [IMAGE_PROMPT] 的中文提示词。
- 必须输出合法的 JSON 代码块。
- 约束条件：{constraints_str}
""".strip()

        user_prompt = f"请开始为【{scene_type}】输出设计与拆解方案。"

        messages = [
            SystemMessage(content="你是一个专业的室内场景规划与拆解助手。严格输出物品清单、布局描述、[IMAGE_PROMPT] 和 JSON。"),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = llm.invoke(messages)
            raw = response.content or ""
            readable_text, structured = _extract_json_block(raw)
            scene_plan_data = structured if isinstance(structured, dict) else None

            
            part = build_part(
                content_type="text",
                content_text=readable_text,  
                parameter={
                    "additional_type": ["scene_plan"],
                    "scene_plan_data": scene_plan_data, 
                    "scene_type": scene_type,
                    "style": style,
                    "final_tool_output": True,
                    "suppress_postprocess": True,
                },
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="text")
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    # ==========================================
    # 3. 注册 StructuredTool
    # ==========================================
    tools = [
        StructuredTool(
            name="generate_product_text",
            description=PRODUCT_TEXT_PROMPTS.tool_description,
            func=_generate_product_text,
            args_schema=ProductTextInput,
        ),
        StructuredTool(
            name="generate_marketing_text",
            description=MARKETING_TEXT_PROMPTS.tool_description,
            func=_generate_marketing_text,
            args_schema=MarketingTextInput,
        ),
        StructuredTool(
            name="generate_creative_text",
            description=CREATIVE_TEXT_PROMPTS.tool_description,
            func=_generate_creative_text,
            args_schema=CreativeTextInput,
        ),
        
        StructuredTool(
            name="generate_scene_plan",
            description="当你需要为一个房间或场景生成【设计方案】、【物品清单】、【布局描述】以及【画图】时调用。警告：调用获取到文本后，你必须立刻提取所有的 [IMAGE_PROMPT] 并发调用 generate_image 生成图片。",
            func=_generate_scene_plan,
            args_schema=ScenePlanInput,
        ),
    ]

    return tools
 
__all__ = ["load_text_tools"]
