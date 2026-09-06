# P4 Agent 交接（2026-09-06）

**实现真值：** [DG-202606-P4-全图导航实现方案.md](./DG-202606-P4-全图导航实现方案.md)（S0–S6）。  
**背景/旧日志：** [DG-202606-P4-导航与全局规划.md](./DG-202606-P4-导航与全局规划.md)。原文混了局部方案、第一版全图和补丁，**不能当现状照做**。  
**几何数字入口：** `runtime/p4_params.py`（裁判框仍来自 `scene_zones.py`）。0.30/0.22/0.18 和 `STOP_DIST=0.43` 是待校准参考，**不是已验收底盘尺寸**。

当前代码还是旧路径（黄线种子、`MAP_READY_HITS`、`_react` 选边、不安全 fallback）。下一任按新方案改，不要把旧行为再写回正式路径。

---

## 直接粘贴给下一任（实现用）

```text
你在 D:\DG\supermarket_sorting，负责 DG-202606 P4 导航实现。
不是文旅 vlm_pipeline，也不是旧路径名 supermarket_sorting_baseline；以本仓 runtime/ 为准。

先读：
1. docs/AGENTS.md
2. docs/DG-202606-AGENT-BRIEF.md
3. docs/DG-202606-P4-AGENT-HANDOFF.md（本文，短）
4. docs/DG-202606-P4-全图导航实现方案.md（实现真值，按 S0–S6）
原 P4 长文只作背景和历史日志，不要把「图够了才 A* / face_turn / 黄线 fallback」再当正式方案。

按新方案 S0→S6 顺序工作，复用 runtime/grid_planner.py，不另建第二套导航。
不要 Nav2 / DWA / 生产插 detour。不要改 client_task_1.py。不要文旅 Task/hug/HSV/DashScope。
障碍只用 /scan，膨胀只算底盘不算手臂。商品坐标不读任务 JSON、不读 Server 真值。
所有速度走 RosRobotController；任何恢复分支都不能绕过安全层。
没被要求不要 git commit。没撞没翻不要重启 Server；撞了才重启。

关键不要违反：
- 启动即允许 A*，不等 MAP_READY_HITS=8。
- 唯一路线写入者是导航状态机；跟随器只消费路径；雷达只否决/限速，不 pick_detour_side。
- 删掉 `_fallback_route(...) or _corridor_seed(...)` 这种绕过碰撞检查的黄线兜底。
- 实体墙 / 中心禁区 / 软走廊分开；不能靠清膨胀或清空目标格开路。
- 全图约束只钉真实墙角；禁止「x>-0.32 就全场禁南行」误杀东通道。
- 首轮仿真用 --speed-scale 1；4× 是速度上限不是仿真时钟，制动标定前不要当验收默认。

入口（先低速）：
  bash scripts/run_p4_nav.sh --goal shelf --speed-scale 1 --timeout 150
  bash scripts/run_p4_nav.sh --goal delivery --speed-scale 1 --timeout 150
日志 outputs/p4/{goal}_{时间}.log/.json 和 last.json。改完 python -m unittest discover -v。

重点回归：
- 20260906 北侧无限 hold：(1.066, 2.486) 被 in_north_racks 零输出
- 20260905 两箱振荡：delivery_20260905_065322
- 不安全 fallback：delivery_20260905_075736 停在 (-1.47, 0.48)
- 黄线南切撞箱：delivery_20260905_103938

成功：到目标并停稳（位置+朝向+区域，不是只进拣货区）；detours=0 且安全层不再选边。
按阶段报告完成内容、测试结果、S0 仍未核定的事实。分别写清单测和仿真，不要把 unittest 写成仿真已过。
```

---

## 新方案必须先记住的决定

1. **启动即 A***。空地图也能规划；unknown 可搜但执行只走可刹停的已验证前段。
2. **探索是带目标的 DFS 分支记录**（红/绿绕行：前方命中 ≠ 整条不通）。`EXPLORE` 选观察子目标，主目标到达才报 `arrived`。
3. **状态机唯一出速度**：`WAIT_SENSORS / PLAN / FOLLOW / BRAKE_WAIT / OBSERVE / RECOVER / EXPLORE / ALIGN_GOAL / ARRIVED / BLOCKED`。禁止无限 `hold`。
4. **本局地图常驻**。`set_goal()` 只换目标清路线，不清图。
5. **目标用 `NavGoal`**，不要再用 `goal_xy[0] < 0` 区分货架/配送。
6. **S0 先核定**：底盘包络、固定结构物理边界、雷达到底盘外参、时钟与 scan 无回波语义。核不定的数不能靠放宽常量假装安全。

阶段：S0 几何与输入 → S1 地图 → S2 规划验证 → S3 跟随与门控 → S4 状态机 → S5 日志/回归 → S6 多目标会话。

---

## 场地区块（世界 +X 东 +Y 北）

数字以 `runtime/p4_params.py` 为准。S0 要标注每项是实体、中心禁区还是软偏好；`NORTH_RACK_Y=2.48` 与 `SHELF_APPROACH_XY.y=2.475` 只差 5 mm，**先解释含义冲突，不要 carve 出通路**。

| 名字 | 常量 | 范围 / 点 |
| --- | --- | --- |
| 出生点 | `SPAWN_XY` | `(1.92, -3.17)` 朝北 |
| 出生范围 | `SPAWN_ZONE` | `x∈[1.70, 2.20]` `y∈[-3.45, -2.90]` |
| 货架本体 | `SHELF_BODY_ZONE` | `y ≥ 2.48`（校准前当行为阈值） |
| 拣货走廊 | `PICKING_AISLE_ZONE` | `y∈[1.70, 3.25]` |
| 货架接近 | `SHELF_APPROACH_XY` | `(0.852, 2.475)` |
| 东通道 | `EAST_CORRIDOR_ZONE` | 隔墙以东；**允许正常南行** |
| 西走廊 | `WEST_CORRIDOR_ZONE` | `x ≤ -0.40`，软优先内轨 |
| 中间隔墙 | `CENTER_WALL_ZONE` | `x∈[0.20, 1.45]` `y∈[-3.08, 1.90]` 禁止西穿 |
| 南缝 | 中心禁区，**双向** | 不禁止西廊去配送、不禁止东廊南行 |
| 配送接近 | `DELIVERY_APPROACH_XY` | `(-1.940, -2.90)`，须 `in_delivery_base` |
| 导航外包 | `NAV_BOUNDS` | 搜索范围，不等于物理外墙 |

无箱软走廊（代价带，不是硬墙）：东上 `(1.92, 2.32)` → 西到货架；回程 `(-0.40, 2.32)` → `(-0.40, -2.70)` → `(-1.94, -2.90)`。

---

## 已作废的正式路径（实现时删掉或降为兼容测试）

- 等确认格 ≥8 才 A*
- 生产 `_react` / `face_turn` / `nudge` / `pick_detour_side`
- 无条件黄线 `_corridor_seed` fallback
- `early_southbound` 全场禁南
- 无限 `hold`（守卫命中必须有原因事件）
- 默认 4× 当第一版验收

旧 `CorridorFollower` 单测可留；`test_runtime_grid_planner.py` 里锁死 `replan corridor` / `explore` / `face_turn` 的断言随契约改掉。

---

## 最近日志（回归用例，不是算法说明书）

| 日志 | 现象 | 新方案应对 |
| --- | --- | --- |
| `shelf/delivery_20260906_07*` | `(1.066, 2.486)` 持续 hold，`plan_id=1` | 禁区守卫 → 事件 + 安全退出，不能零输出卡死 |
| `delivery_20260905_065322` | 两箱左右对拧 | 一图记两箱，整段重搜，安全层不选边 |
| `delivery_20260905_075736` | 西侧箱群 + 两点 fallback | 无解分类恢复，禁止未验证黄线 |
| `delivery_20260905_103938` | 黄线南切后撞箱 | 启动即 A* + 只执行可刹停前段，不抄未验证对角 |
