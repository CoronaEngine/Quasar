"""
Agent 执行模块
负责 Agent 的创建、运行和备用完成逻辑

并发安全说明：
- Agent 本身是无状态的（状态在 messages 中传递）
- 使用 RLock 保护 Agent 创建过程
- 支持多用户并发调用

【当前架构】：支持“拆分-聚合 (Scatter-Gather)”多场景并发图文生成模式
- 已修复 AIMessage 导入问题
- 已实现同步等待图片真实 URL 与前端 HTML <img /> 渲染支持
- 已实现“一个风格一个独立气泡”的渲染格式
"""

from __future__ import annotations

import json
import logging
import threading
import time
import concurrent.futures
from typing import Any, Dict, List

# 👇 修复了你截图中报的 AIMessage 未导入问题
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call

from ai_config.ai_config import AIConfig, get_ai_config
from ai_models.base_pool import get_chat_model
from ai_tools.registry import get_tool_registry


logger = logging.getLogger(__name__)

_CACHED_AGENT: Any = None
_AGENT_LOCK = threading.RLock()


def _should_retry(error: Exception) -> bool:
    
    if isinstance(error, TypeError):
        error_msg = str(error).lower()
        if "choices" in error_msg or "null" in error_msg:
            return True


    error_msg = str(error).lower()
    retryable_keywords = [
        "timeout", "rate limit", "429", "503", "502", "connection", "temporary",
    ]
    return any(keyword in error_msg for keyword in retryable_keywords)


@wrap_model_call
def _retry_middleware(request, handler):
    
    max_retries = 2
    initial_delay = 1.0
    backoff_factor = 2.0

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries and _should_retry(e):
                delay = initial_delay * (backoff_factor**attempt)
                logger.warning(
                    f"模型调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，"
                    f"{delay:.1f}秒后重试..."
                )
                time.sleep(delay)
            else:
                raise
            
    raise last_error  # type: ignore


def _build_agent(config: AIConfig) -> Any:
    
    chat_cfg = config.chat
    
    llm = get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )


    registry = get_tool_registry()
    if not registry.list_tools():
        from ai_tools.load_tools import load_tools
        load_tools(config)

    tools = registry.list_tools()
    logger.debug(f"Agent 使用 {len(tools)} 个工具: {[t.name for t in tools]}")

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=chat_cfg.system_prompt,
        middleware=[_retry_middleware],
    )


def create_default_agent(force_reload: bool = False) -> Any:
    
    global _CACHED_AGENT
    
    if _CACHED_AGENT is None or force_reload:
        
        with _AGENT_LOCK:
            if _CACHED_AGENT is None or force_reload:
                _CACHED_AGENT = _build_agent(get_ai_config())
                
    return _CACHED_AGENT


def _is_connection_error(error: Exception) -> bool:
    
    error_msg = str(error).lower()
    connection_keywords = ["connection", "timeout", "unreachable", "refused"]
    return any(keyword in error_msg for keyword in connection_keywords)


# ============================================================================
# 并发多场景生成引擎 (同步死等与 URL 精洗隐私版)
# ============================================================================

def _run_single_scene_pipeline(scene_style: str, user_input: str) -> str:
    """【子线程】：执行 1 个指定风格的同步图文拆解工作流"""
    logger.info(f"🚀 [子线程] 开始执行串行线路：[{scene_style}]")
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    
    llm = get_chat_model(
        provider_name=chat_cfg.provider, model_name=chat_cfg.model,
        temperature=0.6, request_timeout=chat_cfg.request_timeout,
    )
    
    system_prompt = f"""你是一个高级室内设计师。
用户的原始需求是：【{user_input}】。请专门针对【{scene_style}】风格提供设计方案。

【排版与语气要求】：
1. 语言要优雅、专业、有画面感，像家居杂志的文案。
2. 绝对不要在物品名称前后使用方括号[]等奇怪的符号。

请严格按照以下结构输出可读文本：

**🛋️ 核心单品**
- 物品名称1（直接写名字，不要加括号）
- 物品名称2
- 物品名称3

**📐 空间布局**
- 物品名称1：描述它在空间中的最佳位置和搭配建议。
- 物品名称2：描述它在空间中的最佳位置和搭配建议。

**3. 单体物品画图提示词**：
(为上述单品写纯英文高质量AI绘画Prompt，包含主体、风格、纯白背景等)
[IMAGE_PROMPT] A single English description of object 1, {scene_style} interior design style, isolated on pure white background, studio lighting, 3d render, octane render, photorealistic, masterpieces
[IMAGE_PROMPT] A single English description of object 2, {scene_style} style, isolated on clean white background, product photography, soft cinematic lighting, masterpieces
"""
    
    # ====== 1. 文本生成节点 ======
    try:
        text_msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content="请开始输出该风格的方案。")])
        scene_text = str(text_msg.content)
    except Exception as e:
        logger.error(f"[{scene_style}] 文本生成失败: {e}")
        return f"### 💡 风格方案：{scene_style}\n方案生成致命错误。"
    
    # ====== 2. 提取 Prompt 节点 ======
    image_prompts = []
    for line in scene_text.split('\n'):
        if "[IMAGE_PROMPT]" in line:
            prompt = line.replace("[IMAGE_PROMPT]", "").strip()
            if prompt:
                image_prompts.append(prompt)
                
# ====== 3. 画图节点 (极速异步提交，不阻塞) ======
    registry = get_tool_registry()
    if not registry.list_tools():
        from ai_tools.load_tools import load_tools
        load_tools(get_ai_config())
        
    tools_map = {t.name: t for t in registry.list_tools()}
    image_tool = tools_map.get("generate_image") 
    
    def _map_internal_file_id_to_http(internal_url):
        if not internal_url or not internal_url.startswith("fileid://"):
            return internal_url
        file_hash = internal_url.replace("fileid://", "")
        # 将 fileid 映射为可直连的代理地址
        return f"/api/v1/resource/get?id={file_hash}" 

    image_results = []
    if image_tool and image_prompts:
        for img_prompt in image_prompts:
            try:
                logger.info(f"🎨 [{scene_style}] 正在极速提交画图任务... {img_prompt[:20]}...")
                
                # 仅仅提交一次任务，拿到凭证 JSON
                submit_result_raw = image_tool.invoke({"prompt": img_prompt})
                
                extracted_file_id = ""
                try:
                    # 精准挖出原生的 fileid://...
                    parsed_s = json.loads(submit_result_raw) if isinstance(submit_result_raw, str) else submit_result_raw
                    extracted_file_id = parsed_s["llm_content"][0]["part"][0]["content_url"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                    logger.error(f"[{scene_style}] 解析结果失败: {e}")
                    extracted_file_id = str(submit_result_raw)
                
                # 如果没提取出干净的 fileid，直接过滤掉，防止乱码外溢
                if extracted_file_id.count("{") > 1:
                     extracted_file_id = "" 
                
                if extracted_file_id:
                    # 👇 核心：完全回归你系统的原生格式！让编辑器前端直接识别
                    image_results.append(f"![{scene_style}单品]({extracted_file_id})")
                
            except Exception as e:
                logger.error(f"[{scene_style}] 画图流程异常: {e}")
    else:
        logger.warning(f"[{scene_style}] 未找到 generate_image 工具或未生成 Prompt。")

    # ====== 4. 组装与隐私过滤 ======
    clean_scene_text_lines = []
    for line in scene_text.split('\n'):
        # 彻底隐藏英文 Prompt 和关联标题
        if "[IMAGE_PROMPT]" not in line and "单体物品画图提示词" not in line and "####" not in line and "***" not in line:
            clean_scene_text_lines.append(line)
            
    clean_scene_text = "\n".join(clean_scene_text_lines).strip()

    # 聚合干净完美的图文 Markdown 报告
    final_output = f"### ✨ {scene_style}\n\n{clean_scene_text}\n\n**🖼️ 单品预览**：\n"
    
    if image_results:
        # 直接用换行符拼接原生 Markdown，不要套任何 HTML 标签
        final_output += "\n".join(image_results)
    else:
        final_output += "（无图片生成）"
        
    logger.info(f"✅ [子线程线路完成]：[{scene_style}]")
    return final_output


def execute_multi_scene_parallel(user_input: str, style_count: int = 3) -> List[str]:
    """【主线程】：拆分风格，并返回 3 个独立的文本气泡内容"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    llm = get_chat_model(
        provider_name=chat_cfg.provider, model_name=chat_cfg.model,
        temperature=0.7, request_timeout=chat_cfg.request_timeout,
    )
    
    split_prompt = f"""用户输入了设计需求：【{user_input}】。
请构思 {style_count} 种截然不同的室内设计风格名称。
直接输出JSON数组，例如：["现代极简风", "赛博朋克风", "轻奢法式风"]。不要输出任何解释。"""
    
    logger.info("🧠 多场景引擎：正在拆分设计风格...")
    split_msg = llm.invoke([HumanMessage(content=split_prompt)])
    
    try:
        json_str = split_msg.content.strip()
        if json_str.startswith("```json"): json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"): json_str = json_str[3:-3].strip()
        styles = json.loads(json_str)
        if not isinstance(styles, list): raise ValueError
    except Exception:
        styles = ["现代极简风", "轻奢法式风", "日式原木风"][:style_count]
        
    logger.info(f"⚡ 准备并发执行 {len(styles)} 条线路 -> {styles}")

    # 使用列表保存各个风格的文本
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(styles)) as executor:
        future_to_style = {executor.submit(_run_single_scene_pipeline, style, user_input): style for style in styles}
        for future in concurrent.futures.as_completed(future_to_style):
            try:
                # 把每一条子线程跑出来的完整图文（属于某一个风格），单独 append 进去
                results.append(future.result())
            except Exception as exc:
                results.append(f"### 💡 风格方案：{future_to_style[future]}\n执行过程产生异常。")

    # 👇 核心修改：返回字符串列表（List），不再把它们 join 拼接成一整块！
    return results

# ============================================================================


def run_agent(messages: List[BaseMessage]) -> Dict[str, Any]:
    """运行 agent 入口 (同步)"""
    user_input = str(messages[-1].content)
    
    # 🎯 并发意图拦截
    keywords = ["设计", "风格", "房间", "场景", "拆解", "卧室", "客厅", "电竞房"]
    if any(kw in user_input for kw in keywords):
        try:
            logger.info("🎯 命中场景生成意图，路由至多线程并发引擎...")
            
            # final_texts 是一个包含 3 个字符串的 List
            final_texts = execute_multi_scene_parallel(user_input, style_count=3)
            
            # 👇 核心修改：将 3 个文本分别包装成 3 个独立的 AIMessage，前端会渲染成 3 个气泡框！
            ai_messages = [AIMessage(content=text) for text in final_texts]
            return {"messages": messages + ai_messages}
            
        except Exception as e:
            logger.error(f"并发多场景生成失败: {e}，回退到普通 Agent 模式")

    # 未命中则走原普通 Agent 逻辑
    max_account_switches = 2  
    last_error = None
    
    for switch_attempt in range(max_account_switches + 1):
        try:
            
            agent = create_default_agent(force_reload=(switch_attempt > 0))
            return agent.invoke({"messages": messages})
        
        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                logger.warning(f"连接错误重试 ({switch_attempt + 1}): {e}")
                time.sleep(1.0) 
            else:
                raise
            
    if last_error:
        
        raise last_error
    return {"messages": []}


def stream_agent(messages: List[BaseMessage]):
    """流式运行 agent 入口 (Stream)"""
    user_input = str(messages[-1].content)
    
    # 🎯 并发意图拦截 (Stream 也要拦截)
    keywords = ["设计", "风格", "房间", "场景", "拆解", "卧室", "客厅", "电竞房"]
    if any(kw in user_input for kw in keywords):
        try:
            logger.info("🎯 [Stream入口] 命中多场景意图，路由至并发引擎...")
            
            # final_texts 是一个包含 3 个字符串的 List
            final_texts = execute_multi_scene_parallel(user_input, style_count=3)
            
            # 👇 核心修改：同样包装成 3 个气泡框推流出去
            ai_messages = [AIMessage(content=text) for text in final_texts]
            
            # 直接返回多个气泡节点
            yield {"agent": {"messages": ai_messages}}
            return  
            
        except Exception as e:
            logger.error(f"并发多场景流式生成失败: {e}，回退到普通 Agent 模式")

    # 未命中走原来的原生流式逻辑
    max_account_switches = 2  
    last_error = None
    
    for switch_attempt in range(max_account_switches + 1):
        try:
            
            agent = create_default_agent(force_reload=(switch_attempt > 0))
            
            for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
                
                yield chunk
                
            return
        
        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                time.sleep(1.0)  
            else:
                raise
            
    if last_error:
        raise last_error


def fallback_completion(history: List[BaseMessage]) -> str:
    """备用完成方法"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    
    llm = get_chat_model(
        provider_name=chat_cfg.provider, model_name=chat_cfg.model,
        temperature=chat_cfg.temperature, request_timeout=chat_cfg.request_timeout,
    )
    prompt_messages: List[BaseMessage] = [SystemMessage(content=chat_cfg.system_prompt), *history]
    ai_message = llm.invoke(prompt_messages)
    content = ai_message.content or ""
    
    if isinstance(content, list):
        content = "\n".join([b["text"] for b in content if b.get("type") == "text"])
        
    return content


__all__ = ["create_default_agent", "run_agent", "stream_agent", "fallback_completion"]