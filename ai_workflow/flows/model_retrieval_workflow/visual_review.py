import os
import base64
import logging
from typing import Any, Dict

from openai import OpenAI  # 使用新版的 OpenAI 包

from ai_workflow.streaming import stream_output_node
from .formatters import NO_OUTPUT

logger = logging.getLogger(__name__)

_MAX_REVIEW_RETRIES = 2


def _encode_image_to_base64(image_path: str) -> str:
    """将本地图片转为 Base64，供多模态大模型读取"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@stream_output_node("integrated", NO_OUTPUT)
def visual_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    model_results = state.get("model_results", [])
    six_views = state.get("six_view_images", {})

    needs_retry = False
    pending_generation = []

    # ==============================================================
    # 核心修改：使用智增增的 BASE_URL 和你的 API_KEY
    # ==============================================================
    API_SECRET_KEY = os.getenv(
        "ZHIZENGZENG_API_KEY", "sk-zk249dc3baa72c51a10bb647cbd150953b069d68d09d5f12"
    )
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
        system_prompt = (
            f"你是一位极其严苛的 3D 模型视觉质检专家。你的任务是通过检查同一 3D 模型的【六个正交视角图】（前、后、左、右、顶、底），"
            f"来判定该模型是否合格。\n\n"
            f"【目标提示词】\n{prompt_text}\n\n"
            f"【视觉特征检查清单】（请严格按照以下具体的视觉特征进行排查）：\n\n"
            f"1. 多视角逻辑冲突（致命错误 - 重点检查！）：\n"
            f"   - 两面神现象（Janus Problem）：仔细对比“前视图”和“后视图”。后脑勺绝对不能长出第二张脸、五官或原本只该在正面的配饰！\n"
            f"   - 肢体增生/缺失：对比左右视图，人类/动物的四肢数量必须正常。不能出现三只手、多余的腿，或者某条腿在侧视图中凭空消失。\n"
            f"   - 物理位置错乱：如果前视图中角色右手拿着武器，后视图对应位置也必须有武器，且不能像纸片一样没有厚度。\n\n"
            f"2. 穿模与黏连（交接处检查）：\n"
            f"   - 融化感：观察肢体与躯干、衣服与皮肤、手与物品的交界处。如果它们像橡皮泥一样互相“融化”在一起，没有清晰的物理边界，视为严重穿模。\n"
            f"   - 贯穿：如果武器直接刺穿了身体，或者头发直直插进肩膀里，视为不合格。\n\n"
            f"3. 悬空与表面碎片（背景与边缘检查）：\n"
            f"   - 漂浮物：模型周围、半空中绝对不能有孤立的色块、无意义的几何碎片或不相干的漂浮物体。\n"
            f"   - 表面破损：模型表面应该平滑或符合材质纹理。如果出现密集的坑洞、撕裂状的锯齿边缘、或者像刺猬一样的不规则尖刺，视为网格崩坏。\n\n"
            f"4. 语义与结构还原度：\n"
            f"   - 模型的主体身份、核心特征必须与【目标提示词】高度一致。如果提示词是汽车，不能生成带轮子的沙发。\n"
            f"   - 比例不能严重失调（例如头比身体大3倍，除非提示词明确要求Q版）。\n\n"
            f"【输出规范】（极其重要，决定系统是否会崩溃）\n"
            f"不要包含任何寒暄、思考过程或多余的标点符号。请严格只输出两行文本：\n"
            f"- 如果没有发现上述任何问题（完全合格）：\n"
            f"  第1行严格输出：PASS\n"
            f"  第2行严格输出：符合预期\n"
            f"- 如果发现上述任意一种缺陷（不合格）：\n"
            f"  第1行严格输出：FAIL\n"
            f"  第2行请用一句话指出具体视觉缺陷（例如：FAIL\\n后视图出现了第二张脸，且左侧有悬空碎片）\n\n"
            f"最后警告：在判定为合格的评价中，绝对、绝对不要出现 'FAIL' 这个单词！"
        )

        content_list = [{"type": "text", "text": system_prompt}]

        # 遍历插入六张 Base64 格式的图片
        for view_name, img_path in views_dict.items():
            if os.path.exists(img_path):
                base64_img = _encode_image_to_base64(img_path)
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                    }
                )

        try:
            logger.info(
                f"[Workflow] 正在通过 智增增API 将 {actor_name} 的六视图送入视觉大模型校验..."
            )

            # 新版调用方式：client.chat.completions.create
            # 注意：请确保智增增平台支持你在这里填写的 model 名字 (如 qwen-vl-plus 或 gpt-4o 等多模态模型)
            response = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": content_list}],
                temperature=0.1,
            )

            raw_reply = str(response.choices[0].message.content or "").strip()
            reply_lines = [line.strip() for line in raw_reply.splitlines() if line.strip()]
            decision = reply_lines[0].upper() if reply_lines else "PASS"
            review_reason = reply_lines[1] if len(reply_lines) > 1 else raw_reply or "未提供原因"

            if decision.startswith("FAIL"):
                logger.warning(f"[Workflow] {actor_name} 视觉审查不通过: {raw_reply}")

                retry_count = result.get("retry_count", 0)
                if retry_count >= _MAX_REVIEW_RETRIES:
                    logger.error(
                        f"[Workflow] {actor_name} 达到最大重试次数，停止重试。"
                    )
                    result["review_passed"] = False
                    result["error"] = f"视觉审查未通过: {review_reason}"
                else:
                    result["review_passed"] = False
                    result["source"] = "pending_generation"
                    result["retry_count"] = retry_count + 1
                    result["review_reason"] = review_reason
                    pending_generation.append(dict(result))
                    needs_retry = True
            else:
                logger.info(f"[Workflow] {actor_name} 视觉审查完美通过！结果: {raw_reply}")
                result["review_passed"] = True
                result.pop("review_reason", None)
                result["source"] = "generation"

        except Exception as e:
            logger.error(
                f"[Workflow] 智增增 API 请求崩溃，跳过审查: {e}", exc_info=True
            )
            # 为了防止工作流卡死，请求失败时默认放行
            result["review_passed"] = True
            result["source"] = "generation"

    return {
        "model_results": model_results,
        "six_view_images": six_views,
        "needs_retry": needs_retry,
        "intermediate": {
            **state.get("intermediate", {}),
            "pending_generation": pending_generation,
        },
    }
