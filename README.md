# CoronaArtificialIntelligence

AI模块加载，根据ai功能划分模块并通过配置实现动态加载

---

- 1、项目根目录/ai_service/entrance.py
  - 初始化ai_entrance类
  - 配置文件: 项目根目录/service/module_settings.yaml
    - 根据yaml顺序加载modules目录下的模块
      - name —— 模块名
      - enabled —— 是否加载
      - description —— 描述

---

- 2、ai_modules 模块文件结构
  - 根目录/
    - configs/
      - dataclasses.py config基类
      - prompts.py ai提示词
      - settings.py ai配置
    - tools
      - loader.py config生成器
      - 。。。其他工具
    - base.py 功能接口

---

- 3、模块收集装饰器 ConfigCollector
  - settings配置
    - 保存ai设置
    - @ai_entrance.collector.register_setting(str)
  - loader生成
    - 获取保存的ai设置并通过函数将config集成进AIConfig中(需要注册settings)
    - @ai_entrance.collector.register_loader(str)

---

- 4、ai_entrance使用
  - register_entrance接口注册
    - 将模块函数注册进ai_entrance对象中
    - @register_entrance(handler_name=func_name)
  - 接口使用
    - @register_entrance(handler_name="handle_text_generation")
      def handle_text_generation(payload: Any)
    - ai_entrance().handle_text_generation(payload)

---

## 其他模块

- ai_agent/ 会话存储
- ai_media_resource/ 文件处理
- ai_workflow/ 工作流
- ai_models/
- ai_config/
- ai_tools/ 工具
