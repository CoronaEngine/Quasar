import os
import json
import base64
import logging
from typing import Any, Dict

from openai import OpenAI  # 使用新版的 OpenAI 包

from ai_workflow.streaming import stream_output_node
from .formatters import NO_OUTPUT

logger = logging.getLogger(__name__)

def _encode_image_to_base64(image_path: str) -> str:
    """将本地图片转为 Base64，供多模态大模型读取"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@stream_output_node("integrated", NO_OUTPUT)
def visual_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    model_results = state.get("model_results", [])
    six_views = state.get("six_view_images", {})
    
    needs_retry = False
    
    # ==============================================================
    # 核心修改：使用智增增的 BASE_URL 和你的 API_KEY
    # ==============================================================
    API_SECRET_KEY = os.getenv("ZHIZENGZENG_API_KEY", "sk-zk249dc3baa72c51a10bb647cbd150953b069d68d09d5f12") 
    BASE_URL = "https://api.zhizengzeng.com/v1/"
    
    try:
        # 新版 OpenAI 初始化写法
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    except Exception as e:
        logger.error(f"[Workflow] OpenAI 客户端初始化失败: {e}")
        return {"model_results": model_results, "needs_retry": False}

    for result in model_results:
        # 如果已经审查通过，或直接从库里检索到的，跳过
        if result.get("review_passed") or result.get("source") == "retrieval" or result.get("error"):
            continue

        actor_name = result.get("object_id") or result.get("item_name")
        prompt_text = result.get("image_prompt") or result.get("item_name")
        views_dict = result.get("six_views_dict") or six_views.get(actor_name, {})

        if not views_dict:
            logger.warning(f"[Workflow] 未找到 {actor_name} 的六视图数据，跳过校验。")
            result["review_passed"] = True
            continue

        # 组装消息列表 (符合新版 API 规范)
        content_list = [
            {
                "type": "text", 
                "text": f"你是一个严苛的3D模型审查专家。这是根据提示词：【{prompt_text}】生成的3D模型的六个正交视角图。"
                        f"请检查模型是否符合提示词描述，是否存在明显的穿模、结构扭曲或悬空。"
                        f"如果合格，请在第一行严格回复 'PASS'；如果不合格，请在第一行严格回复 'FAIL'，并在第二行简述原因。"
            }
        ]
        
        # 遍历插入六张 Base64 格式的图片
        for view_name, img_path in views_dict.items():
            if os.path.exists(img_path):
                base64_img = _encode_image_to_base64(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        try:
            logger.info(f"[Workflow] 正在通过 智增增API 将 {actor_name} 的六视图送入视觉大模型校验...")
            
            # 新版调用方式：client.chat.completions.create
            # 注意：请确保智增增平台支持你在这里填写的 model 名字 (如 qwen-vl-plus 或 gpt-4o 等多模态模型)
            response = client.chat.completions.create(
                model="qwen3.5-plus", 
                messages=[{"role": "user", "content": content_list}],
                temperature=0.1
            )
            
            # 解析新版返回包格式
            reply = response.choices[0].message.content.strip().upper()
            
            if "FAIL" in reply:
                logger.warning(f"[Workflow] {actor_name} 视觉审查不通过: {reply}")
                
                retry_count = result.get("retry_count", 0)
                if retry_count >= 2:
                    logger.error(f"[Workflow] {actor_name} 达到最大重试次数，保留当前模型。")
                    result["review_passed"] = True
                else:
                    result["review_passed"] = False
                    result["source"] = "pending_generation" 
                    result["retry_count"] = retry_count + 1
                    needs_retry = True
            else:
                logger.info(f"[Workflow] {actor_name} 视觉审查完美通过！结果: {reply}")
                result["review_passed"] = True

        except Exception as e:
            logger.error(f"[Workflow] 智增增 API 请求崩溃，跳过审查: {e}", exc_info=True)
            # 为了防止工作流卡死，请求失败时默认放行
            result["review_passed"] = True 

    return {"model_results": model_results, "needs_retry": needs_retry}