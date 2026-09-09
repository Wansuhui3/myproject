"""回调包：按功能域拆分的回调模块。

唯一连接 core/、comparison/ 和 components/ 的桥梁。
模块依赖方向（避免循环导入）：
  wave_helpers / wave_views          → 纯数据 / 纯渲染（不导入回调模块）
  wave_upload_callbacks              → wave_helpers, wave_views
  wave_callbacks                     → wave_helpers, wave_views
  cmp_preview_render                 → 纯渲染（不导入回调模块）
  cmp_mode / cmp_upload / cmp_mapping_callbacks → cmp_preview_render
  comparison_callbacks（C3/C5）      → cmp_preview_render
  performance_callbacks              → helpers + cmp_mapping_callbacks
  clientside_callbacks               → extensions.app

回调清单：
波动分析（wave_upload_callbacks）：
  on_radar_change_label / on_radar_change_clear → 雷达切换 → 更新位置标签 / 清空页面
  on_upload_csv            → 拖拽上传 → 解析CSV/JSON → 分段 → 缓存
  on_timestamp_input       → 筛选时间窗口 → 提取ID列表
  refresh_quantity_options → 数据加载后刷新物理量多选项
  on_wave_clear            → 清空波动分析页面
波动分析（wave_callbacks）：
  on_id_click              → 自定义 ListGroup 点击选中
  on_trajectory_select     → 轨迹段选中 → 绘制多子图 + 全段统计
  on_quantity_change       → 物理量多选变更 → 重建子图 + 重算统计
  on_box_select            → 框选 → 全子图高亮 + 统计
  on_clear_box_select      → 清除框选
  update_wave_summary_snapshots / render_wave_summary_content
                           → 波动摘要快照插入/移除/渲染

真值对比（cmp_mode_callbacks）：
  on_cmp_clear             → 清空对比页面与对比缓存
  [C1] on_mode_switch      → 模式切换 → 显示/隐藏面板 + 重置对比状态
真值对比（cmp_upload_callbacks）：
  [C2] on_cmp_upload       → 统一上传 → 解析+缓存+预览+ID发现（单回调无轮询）
真值对比（cmp_mapping_callbacks）：
  [C2a] render_cmp_preview_summary / on_cmp_mapping_sources /
       render_cmp_mappings / on_cmp_mapping_edit
                           → 原始CSV物理量映射配置（预览/增删改/排序）
真值对比（comparison_callbacks）：
  [C3] on_cmp_id_select    → ID选择 → 坐标诊断 + 延迟检测
  [C5] on_cmp_mapping_change → 映射变更提示

对齐与性能验收（performance_callbacks）：
  [C4] update_selected_segments / on_cmp_run
                           → 段勾选变更；执行对齐 → 图表 + 统计 + 性能验收
  [C4a] update_perf_summary_snapshots / render_perf_summary_snapshots
                           → 分距离性能摘要快照
  [C4b] on_perf_metric_switch → 性能验收指标切换
  [C4c] on_perf_collapse   → 性能验收折叠切换
  [C6] on_cmp_export_workbook → 导出对齐+逐帧性能+验收汇总 xlsx 工作簿
"""
# 导入子模块以触发 @callback 注册（app.py 在 app 创建后导入本包）。
from . import (  # noqa: F401
    clientside_callbacks,
    cmp_mapping_callbacks,
    cmp_mode_callbacks,
    cmp_upload_callbacks,
    comparison_callbacks,
    helpers,
    performance_callbacks,
    wave_callbacks,
    wave_upload_callbacks,
)

# 兼容层：保持既有 ``from radar_wave_analyzer.callbacks import X`` 导入路径
# （app.py 与测试套件依赖）。新代码请直接从对应子模块导入。
from .cmp_mapping_callbacks import (  # noqa: F401
    _cmp_mapping_options,
    _cmp_resolve_mappings,
)
from .helpers import _summary_plain_text  # noqa: F401
from .performance_callbacks import _selected_radar_filename  # noqa: F401
from .wave_helpers import (  # noqa: F401
    _discover_quantity_columns,
    _ensure_valid_quantities,
)
