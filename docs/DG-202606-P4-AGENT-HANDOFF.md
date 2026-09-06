# P4 Agent 交接（2026-09-06）

把下面整段（或本文）交给下一任即可。数字以 `runtime/p4_params.py` 为准，不要再手抄一套。

---

## 直接粘贴给下一任

```text
你在 DG-202606 智慧零售仓 D:\DG\supermarket_sorting（不是文旅 vlm_pipeline，
也不是旧路径名 supermarket_sorting_baseline，以本仓 runtime/ 为准）。

先读：
1. docs/AGENTS.md
2. docs/DG-202606-AGENT-BRIEF.md
3. 本文 docs/DG-202606-P4-AGENT-HANDOFF.md
4. 细节不够再翻 docs/DG-202606-P4-导航与全局规划.md

当前只做 P4：货架接近点 ↔ 配送接近点。不要 Nav2 / DWA / 插 detour。
不要改 client_task_1.py，不要文旅 Task/hug/HSV/DashScope。
障碍只用 /scan，膨胀只算底盘不算手臂。
改数只改 runtime/p4_params.py。改完 python -m unittest discover -v。
没被要求不要 git commit。没撞没翻不要重启 Server；撞了才重启。

入口：
  bash scripts/run_p4_nav.sh --goal shelf
  bash scripts/run_p4_nav.sh --goal delivery
默认 4×。日志 outputs/p4/{goal}_{时间}.log/.json 和 outputs/p4/last.json。

当前策略（已落地，不要推倒重来）：
- 第一轮图不够：跟黄线走廊，用雷达 clear / nudge / approach / face_turn。
  正前方清空距离 = stop_dist + 0.30×v。挡得窄或远且一侧空 → 边走边修；
  挡得满 → 先靠过去，贴停再原地转。不插绕行点。
- 雷达确认大约一块箱子（≥ MAP_READY_HITS）后才允许 A*。
- A* 若还是「没贴到 x=-0.40 就南切」，丢掉，继续黄线。
- 开头日志应是 replan corridor，黄线上 plan_id 不该暴涨。

最新失败：delivery_20260905_103938 在黄线 (0.20,2.32) 被空图 A* 带到
隔墙角 (-0.28,1.42)，又以 0.51 m/s 顶着 0.39 m 回波撞箱。
成功标准：--goal delivery 到 in_delivery_base，arrived=true，detours=0。
```

---

## 场地区块（世界 +X 东 +Y 北，米）

权威文件：`runtime/p4_params.py`（裁判框来自 `runtime/scene_zones.py`）。

| 名字 | 常量 | 范围 / 点 |
| --- | --- | --- |
| 出生点 | `SPAWN_XY` / `SPAWN_YAW` | `(1.92, -3.17)` 朝北 |
| 出生范围 | `SPAWN_ZONE` | `x∈[1.70, 2.20]` `y∈[-3.45, -2.90]` |
| 货架本体 A–E | `SHELF_BODY_ZONE` | 全宽，`y ≥ 2.48`（没有各组分开的 MJCF 框） |
| 拣货黄线走廊 | `PICKING_AISLE_ZONE` | `x∈[-2.5, 2.5]` `y∈[1.70, 3.25]` |
| 货架接近 | `SHELF_APPROACH_XY` | `(0.852, 2.475)` |
| 东通道 | `EAST_CORRIDOR_ZONE` | 隔墙以东、货架以南 |
| 西走廊（贴轨） | `WEST_CORRIDOR_ZONE` | `x ≤ -0.40`，`y∈[-2.70, 2.32]` |
| 西侧外车道 | `WEST_BOX_LANE_ZONE` | 箱子常在这，`x ≤ -0.70` |
| 中间隔墙 | `CENTER_WALL_ZONE` | `x∈[0.20, 1.45]` `y∈[-3.08, 1.90]` **禁止西穿** |
| 隔墙西北角 | `DIVIDER_NW_ZONE` | 路径检查，不画成墙 |
| 配送作业区 | `DELIVERY_WORK_ZONE` | `x∈[-2.42, -1.46]` `y∈[-3.88, -2.62]` |
| 桌面放置框 | `DELIVERY_PLACE_ZONE` | 桌面 `z∈[0.74, 1.05]` |
| 配送接近 | `DELIVERY_APPROACH_XY` | `(-1.940, -2.90)`，桌 `(-1.940, -3.410)` |
| 导航外包 | `NAV_BOUNDS` | `x∈[-2.55, 2.35]` `y∈[-3.85, 3.30]` |

无箱推荐走廊：

- 出生 → 货架：东通道北上到 `(1.92, 2.32)` → 西到货架接近点。
- 货架 → 配送：西到 `(-0.40, 2.32)` → 南到 `(-0.40, -2.70)` → 到 `(-1.94, -2.90)`。还没到 `HUG_WEST_X=-0.40` 禁止南切。

通道 5 个随机立箱。官方保证货架入口到配送台入口有通路，不保证黄线全程无箱。

---

## 可调项（同样只改 `p4_params.py`）

| 项 | 常量 | 现在 |
| --- | --- | --- |
| 栅格 | `GRID_RES` | 0.10 m |
| 静态墙膨胀 | `STATIC_INFLATE_M` | 0.30 m |
| 动态箱膨胀 | `INFLATE_M` / `RELAXED_INFLATE_M` | 0.22 / 0.18 m |
| 刹停 | `STOP_DIST` | 0.43 m |
| 清空地平线 | `CLEAR_S` | `stop + 0.30×v` |
| 开 A* 的确认格 | `MAP_READY_HITS` | 8 |
| 默认倍速 | `SPEED_SCALE` | 4 |
| 单测超时 | `DEFAULT_TIMEOUT_S` | 150 s 墙钟 |
| 评分时限 | `TIME_LIMIT_S` | 420 s 仿真钟 |

---

## 代码从哪进

| 文件 | 职责 |
| --- | --- |
| `runtime/p4_params.py` | 几何 + 可调项，**唯一改数处** |
| `runtime/grid_planner.py` | 栅格、A*、`GridNavController`（explore → global） |
| `runtime/lidar_avoid.py` | 扇区、`occlusion_kind`、safety_gate；生产不插 detour |
| `runtime/waypoint_nav.py` | 黄线跟线、`build_*_route` |
| `runtime/p4_nav.py` | ROS 入口 20 Hz |
| `scripts/run_p4_nav.sh` | Client 里启动 |

---

## 最近日志（当回归，不要当轶事）

| 日志 | 结果 |
| --- | --- |
| `shelf_20260905_090049` | 货架 `arrived=true`，`plan_id` 仍偏多 |
| `delivery_20260905_090321` | 西廊外绕到 `x≈-1.8`，超时 |
| `delivery_20260905_103938` | 黄线过早南切，隔墙角撞箱，`plan_id=63` |

下一任先对照 `outputs/p4/last.json` 和最新 `delivery_*.log`，不要重开一套算法。
