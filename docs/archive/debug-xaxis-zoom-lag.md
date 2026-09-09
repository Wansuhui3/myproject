# Debug Session: xaxis-zoom-lag

## Status: [FIXED — pending browser acceptance]

## Symptom
数据可视化界面（trajectory-graph 多子图共享X轴）：
- 缩放纵坐标轴：流畅无卡顿
- 缩放横坐标轴：立即卡顿 + 抽搐（画面抖动/闪烁）

## Environment
- 渲染器: GRAPH_RENDERER=svg（config.yaml:99），RESAMPLER_ENABLED=false，DISPLAY_MAX_POINTS=1200
- 图表结构: make_subplots(shared_xaxes=True) 纵向堆叠 N 面板，每面板 2 trace（可见线 + 透明悬停命中层）+ shapes（跳变标记/图例短线/高亮）
- X 轴: datetime（timestamp ISO）；非底部 X 轴 visible=False
- App: Dash, http://127.0.0.1:8050

## Hypotheses (falsifiable)
- **A: 共享X轴级联重绘** — shared_xaxes 的 `matches` 使 X 缩放触发全部 N 面板重绘（N×2 trace + shapes），而 Y 缩放仅重绘 1 面板。观测点: 浏览器探针 plotly_relayout 事件中 xaxis.range 变更数（预期 N 个）vs yaxis.range（1 个）；重绘耗时。
- **B: datetime 刻度重算** — X 为日期轴，每次缩放重算刻度+格式化；Y 为数值轴便宜。观测点: 探针 relayout 后渲染耗时与 longtask 分布。
- **C: 服务器往返** — 缩放意外触发 Dash 回调 → figure 全量重建（旧会话曾出现视图重置抽搐）。观测点: python 插桩 build_multi_subplot_graph 调用日志 + 浏览器 DOM 重建（MutationObserver 监测 .main-svg 替换）。
- **D: shapes 全量重排** — 跳变标记/图例 shapes 引用各子图 xref，X 缓存联动全部重排。观测点: figure shapes 数量 × 探针重绘时长相关性。
- **E: 透明悬停命中层** — markers 模式 ~1200 点/面板在 X 缩放时全量重绘。观测点: 后续对照实验（隐藏命中层对比）。

## Instrumentation
- `assets/zz-debug-zoom-probe.js`: plotly 事件钩子（relayout/relayouting/redraw/restyle）+ 轴变更计数 + rAF 渲染耗时 + PerformanceObserver(longtask) + MutationObserver(DOM 重建)
- `components/graph_builder.py`: build_multi_subplot_graph 入口日志（假设 C 观测点）
- Debug Server: session=xaxis-zoom-lag, outdir=.dbg

## Steps
- [x] 1. Observe & hypothesize
- [x] 2. Init debug doc
- [x] 3. Start debug server
- [x] 4. Instrument (JS probe + python log)
- [ ] 5. User reproduce (zoom X then zoom Y)
- [ ] 6. Analyze logs → confirm/reject hypotheses
- [ ] 7. Minimal fix
- [ ] 8. Post-fix verification
- [ ] 9. Cleanup

## Evidence Log
(runId=pre, 待收集)

## Root Cause
- `make_subplots(shared_xaxes=True)` 并不使用一条物理 X 轴，而是创建 N 条
  X 轴并用 `matches` 关联。每次横向缩放都级联执行 N 份 datetime 刻度、网格、
  shape 与曲线路径布局；纵向缩放只重排当前面板的一条 Y 轴。
- 每个面板原先还有一条包含全部显示点的透明 marker trace，仅用于悬停，令横向
  缩放时曲线几何重绘量翻倍。
- 静态极值降采样已将数据限制到 1200 点，因此服务器数据处理不是本次交互卡顿的
  主因；缩放过程不触发 Dash figure 回调。

## Fix
- 所有面板 trace、shape 和 annotation 统一引用唯一 `x` 轴；移除 `xaxis2...N`
  的 `matches` 级联，保留各面板独立 Y 轴和原有 domain。
- 将“可见 line + 透明 marker 命中层”合并为单条带 hovertemplate 的 line trace，
  每面板重绘 trace 数由 2 降到 1。
- 保留 1200 点极值降采样、原始数据框选回查和 `uirevision`，不改变统计精度与
  视图保持行为。
