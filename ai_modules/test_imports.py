"""
快速导入检查脚本 - 验证所有模块导入是否正常
"""

import sys
from pathlib import Path

# 设置路径（与 test_recognition.py 相同）
ai_root = (
    Path(__file__).resolve().parents[0]
)  # config 目录本身的 parents[0] 会是 ai_modules
# 修正：ai_modules 就是我们要添加的
ai_modules_dir = Path(__file__).resolve().parent.parent
print(f"AI Modules Dir: {ai_modules_dir}")

for path in (str(ai_modules_dir),):
    if path not in sys.path:
        sys.path.insert(0, path)

print("测试导入...")

try:
    print("1. 导入 ai_config.paths_config...")
    from ai_config.paths_config import get_project_models_dir, get_project_recognition_db

    print(f"   成功! models_dir={get_project_models_dir()}")

    print("\n2. 导入 object_recognition.configs.dataclasses...")
    from object_recognition.configs.dataclasses import RecognitionConfig

    print("   成功!")

    print("\n3. 导入 object_recognition.tools.vector_db...")
    from object_recognition.tools.vector_db import VectorDB, normalize_vector

    print("   成功!")

    print("\n4. 导入 object_recognition.tools.auto_scan...")
    from object_recognition.tools.auto_scan import scan_and_register

    print("   成功!")

    print("\n5. 导入 object_recognition.tools.client_embedding...")
    from object_recognition.tools.client_embedding import Qwen3VLEmbeddingClient

    print("   成功!")

    print("\n✓ 所有导入检查通过！")

except Exception as e:
    print(f"\n✗ 导入失败: {e}")
    import traceback

    traceback.print_exc()
