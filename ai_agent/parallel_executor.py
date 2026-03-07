import concurrent.futures
import logging
import re
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from ai_models.base_pool import get_chat_model
from ai_config.ai_config import get_ai_config
from ai_tools.registry import get_tool_registry

logger = logging.getLogger(__name__)

# ==========================================
# 第一部分：串行线路 (单场景的完整图文生成)
# ==========================================
def _run_single_scene_pipeline(scene_style: str, room_type: str) -> str:
    """
    【单条串行线路】：负责 1 个场景风格的文本生成和图片生成
    """
    logger.info(f"🚀 开始执行串行线路：{room_type} - {scene_style}")
    
    config = get_ai_config()
    llm = get_chat_model(
        provider_name=config.chat.provider,
        model_name=config.chat.model,
        temperature=0.6,
    )
    
    # 1. 文本节点：生成该风格的【物品清单】、【布局】和【画图提示词】
    system_prompt = f"""你是一个资深的室内设计师。当前任务是设计一个【{scene_style}】风格的【{room_type}】。
    
    请严格按照以下格式输出：
    **1. 场景单个物品列表清单**：
    (列出3个核心单体物品)
    
    **2. 单个物体布局的文字描述**：
    (描述这3个物品在空间中的摆放位置和布局关系)
    
    **3. 单体物品画图提示词**：
    (为这3个物品分别写出一句纯中文的AI绘画Prompt)
    [IMAGE_PROMPT] 一个单独的[物品1详细描述]，{scene_style}风格，纯白背景，影棚级别光照，3D渲染，极高画质
    [IMAGE_PROMPT] 一个单独的[物品2详细描述]，{scene_style}风格，纯净白底，专业产品摄影，高清细节
    [IMAGE_PROMPT] 一个单独的[物品3详细描述]，{scene_style}风格，纯白色背景，大师级杰作
    """
    
    # 获取文本生成结果
    text_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始输出该风格的拆解与设计方案。")
    ])
    scene_text = str(text_msg.content)
    
    # 2. 解析节点：提取所有的 [IMAGE_PROMPT]
    image_prompts = []
    for line in scene_text.split('\n'):
        if "[IMAGE_PROMPT]" in line:
            prompt = line.replace("[IMAGE_PROMPT]", "").strip()
            if prompt:
                image_prompts.append(prompt)
                
    # 3. 图像生成节点：调用底层的 generate_image 工具
    registry = get_tool_registry()
    tools = {t.name: t for t in registry.list_tools()}
    image_tool = tools.get("generate_image") # 对应你 image_tools.py 里的工具名
    
    image_results = []
    if image_tool:
        # 在单条串行线内部，如果有多个物品，也可以选择循环画图
        for img_prompt in image_prompts:
            try:
                logger.info(f"🎨 正在作画 [{scene_style}]: {img_prompt[:20]}...")
                # 调用你原有的底层画图工具
                img_url = image_tool.invoke({"prompt": img_prompt})
                image_results.append(f"![{scene_style}物品]({img_url})")
            except Exception as e:
                logger.error(f"画图失败: {e}")
                image_results.append(f"[图片生成失败: {e}]")
    else:
        logger.warning("未找到 generate_image 工具！")

    # 4. 组装单条线路的最终结果
    final_output = f"### 风格方案：{scene_style}\n\n{scene_text}\n\n**生成的效果图**：\n"
    final_output += "\n".join(image_results)
    
    logger.info(f"✅ 串行线路完成：{room_type} - {scene_style}")
    return final_output


# ==========================================
# 第二部分：最外层并发循环 (拆分与线程池)
# ==========================================
def execute_multi_scene_parallel(room_type: str, style_count: int = 3) -> str:
    """
    【主入口】：最外面套的一层并发循环
    """
    config = get_ai_config()
    llm = get_chat_model(
        provider_name=config.chat.provider,
        model_name=config.chat.model,
        temperature=0.7,
    )
    
    # 1. 拆分任务 (Split)：让大模型决定要生成哪几种风格
    split_prompt = f"""用户需要设计一个【{room_type}】。请为它构思 {style_count} 种截然不同的设计风格名称。
    请直接输出一个 JSON 数组，例如：["现代极简", "赛博朋克", "复古法式"]。不要输出其他废话。"""
    
    logger.info("🧠 正在进行任务拆分...")
    split_msg = llm.invoke([HumanMessage(content=split_prompt)])
    
    try:
        # 清理可能存在的 Markdown 代码块包裹
        json_str = split_msg.content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()
            
        styles = json.loads(json_str)
        if not isinstance(styles, list):
            styles = ["现代风格", "科幻风格", "复古风格"][:style_count]
    except Exception as e:
        logger.error(f"风格拆分解析失败: {e}，使用默认风格")
        styles = ["现代极简风", "未来科技风", "自然原木风"][:style_count]
        
    logger.info(f"拆分结果：准备并发执行 {len(styles)} 条线路 -> {styles}")

    # 2. 核心：最外面套一层并发循环 (Parallel Loop)
    results = []
    # 使用 ThreadPoolExecutor 开启多线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(styles)) as executor:
        # 将任务提交给线程池：每个风格跑一条 _run_single_scene_pipeline 串行线
        future_to_style = {executor.submit(_run_single_scene_pipeline, style, room_type): style for style in styles}
        
        for future in concurrent.futures.as_completed(future_to_style):
            style = future_to_style[future]
            try:
                # 获取该条串行线路的执行结果
                res = future.result()
                results.append(res)
            except Exception as exc:
                logger.error(f"线路 [{style}] 执行产生异常: {exc}")
                results.append(f"### 风格方案：{style}\n执行过程中发生错误。")

    # 3. 聚合结果 (Gather)
    final_report = f"## 您的【{room_type}】多方案设计报告\n\n" + "\n\n---\n\n".join(results)
    return final_report