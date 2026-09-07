# P4 Agent 交接（2026-09-07）

**实现真值：** [DG-202606-P4-全图导航实现方案.md](./DG-202606-P4-全图导航实现方案.md)（S0–S6 设计）。  
**现状以本文为准**，不要按方案第 2 节的 9 月 6 日 hold 日志当还没改。  
**背景/旧失败：** [DG-202606-P4-导航与全局规划.md](./DG-202606-P4-导航与全局规划.md)。  
**几何数字入口：** `runtime/p4_params.py`（裁判框仍来自 `scene_zones.py`）。0.30/0.22/0.18 和 `STOP_DIST=0.43` 是待校准参考，**不是已验收底盘尺寸**。

代码仓：`D:\DG\supermarket_sorting`（旧名 `supermarket_sorting_baseline` 已不用）。

---

## 现状（不要从 S0 重写）

S0–S6 已按方案落到 `runtime/grid_planner.py` + `waypoint_nav.py` + `lidar_avoid.py` + `p4_nav.py`。启动即 A*，跟随器只读路线，雷达只刹停/限速，无黄线 unverified fallback，无生产 `pick_detour_side`。

仿真：

| 日志 | 结果 |
| --- | --- |
| `delivery_20260906_093534` | 从箱区 `(-1.82,-0.37)` 进 `in_delivery_base`，150 s 超时。`arrived=false`，还差约 16 cm。密 A* 点把直段刹成 0.11–0.50 m/s。 |
| `delivery_20260906_094432` | 同点再跑，**4.81 s `arrived=true`**，`(-1.937,-2.795)`，`detours=0`，8× 上限。 |

`in_delivery_base` / 看起来靠上桌子 **不等于** `arrived`。到达必须同时：距 `DELIVERY_APPROACH_XY=(-1.94,-2.90)` ≤0.12 m、朝南误差 ≤0.05 rad、在 `delivery_base`、线速 ≤0.02 且保持 0.30 s。桌心是 `(-1.94,-3.41)`，接近点在桌子北侧，底盘不要开进桌肚。

直段速度按**剩余路径**给，不对每个 0.1–0.3 m 栅格点刹车；急弯和最后一段仍刹。已知前段上限 8×=2.40 m/s，探索 4×=1.20，恢复/对准 1×=0.30。不要为了再快而缩小 footprint 或屏蔽雷达。

**还没证明：** 出生点或货架 → 配送的整段新速度日志；多种子随机箱；完整 DFS 探索栈；S0 底盘包络 / 墙面 / `T_base_laser` / 制动标定（`laser=TF_REQUIRED`）。

下一刀不是重写导航，是 **可乐真抓真放**，并补货架往返。不要 Nav2 / DWA / 黄线 fallback。没撞没翻不要重启 Server。

---

## 直接粘贴给下一任

```text
你在 D:\DG\supermarket_sorting，负责 DG-202606。不是文旅 vlm_pipeline。

先读：
1. docs/AGENTS.md
2. docs/DG-202606-AGENT-BRIEF.md
3. docs/DG-202606-P4-AGENT-HANDOFF.md（本文，现状）
4. 抓放再读 docs/DG-202606-B-迁移与算法.md
P4 方案原文只作设计约束，不要按「尚未落地 / 现码仍是旧路径」重写 S0–S6。

P4 已到配送接近点：delivery_20260906_094432 arrived=true。
in_zone 不是 arrived。目标是 (-1.94,-2.90) 朝南，不是桌心 (-1.94,-3.41)。
直段按剩余路径给速；已知 8× / 探索 4× / 恢复 1×。刹停距离不要随便改。

不要 Nav2 / DWA / 生产插 detour / 黄线 unverified fallback。
不要改 client_task_1.py。不要文旅 Task/hug/HSV/DashScope。
障碍只用 /scan。所有速度走 RosRobotController。
没被要求不要 git commit。没撞没翻不要重启 Server。

下一刀：用 kele.pt 做一次真抓真放；需要时再跑
  bash scripts/run_p4_nav.sh --goal shelf --timeout 150
  bash scripts/run_p4_nav.sh --goal delivery --timeout 150
日志 outputs/p4/{goal}_{时间}.log/.json 和 last.json。
改完 python -m unittest discover -v。分别写清单测和仿真。
```

---

## 仍有效的设计决定

1. **启动即 A***。空地图也能规划；unknown 可搜但执行只走可刹停的已验证前段。
2. **探索是带目标的 DFS 分支记录**（前方命中 ≠ 整条不通）。观察子目标完成不报配送 `arrived`。完整栈仍未做完。
3. **状态机唯一出速度**：`WAIT_SENSORS / PLAN / FOLLOW / BRAKE_WAIT / OBSERVE / RECOVER / EXPLORE / ALIGN_GOAL / ARRIVED / BLOCKED`。禁止无限 `hold`。
4. **本局地图常驻**。`set_goal()` 只换目标清路线，不清图。
5. **目标用 `NavGoal` / `goal_kind`**，不要用 `goal_xy[0] < 0` 区分货架/配送。
6. **S0 未核定项不能靠放宽常量假装安全**：底盘包络、固定结构物理边界、雷达到底盘外参、时钟与 scan 无回波语义。

---

## 场地区块（世界 +X 东 +Y 北）

数字以 `runtime/p4_params.py` / `scene_zones.py` 为准。`NORTH_RACK_Y=2.48` 与 `SHELF_APPROACH_XY.y=2.475` 只差 5 mm：行为阈值 vs 命名接近点，**不要 carve 出通路**。

| 名字 | 常量 | 范围 / 点 |
| --- | --- | --- |
| 出生点 | `SPAWN_XY` | `(1.92, -3.17)` 朝北 |
| 货架接近 | `SHELF_APPROACH_XY` | `(0.852, 2.475)` |
| 东通道 | `EAST_CORRIDOR_ZONE` | 隔墙以东；**允许正常南行** |
| 西走廊 | `WEST_CORRIDOR_ZONE` | `x ≤ -0.40`，软优先内轨 `HUG_WEST_X` |
| 西行黄线 | `WEST_LANE_Y` | `2.32`；东通道须先北上再西转 |
| 中间隔墙 | `CENTER_WALL_ZONE` | `x∈[0.20, 1.45]` `y∈[-3.08, 1.90]` 禁止西穿 |
| 配送作业区 | `DELIVERY_BASE_ZONE` | `x∈[-2.42,-1.46]` `y∈[-3.88,-2.62]` |
| 配送接近 | `DELIVERY_APPROACH_XY` | `(-1.940, -2.90)` 朝南，须 `in_delivery_base` |
| 配送桌 | `DELIVERY_TABLE_XY` | `(-1.940, -3.410)`，放货高度约 0.807 m |

无箱软走廊：东上 `(1.92, 2.32)` → 西到货架；回程 `(-0.40, 2.32)` → `(-0.40, -2.70)` → `(-1.94, -2.90)`。

---

## 已作废的正式路径（不要写回）

- 等确认格 ≥8 才 A*
- 生产 `_react` / `face_turn` / `nudge` / `pick_detour_side`
- 无条件黄线 `_corridor_seed` fallback
- `early_southbound` 全场禁南
- 无限 `hold`（守卫命中必须有原因事件）
- 车心未进实体墙就把膨胀/墙回波判成 `START_IN_COLLISION` 死停
- 同伦两条路来回对切（`ROUTE_HOLD_S` 期间不换边；夹角 >2 rad 视为回头）
- 鼻子顶箱还不掉头、A* 失败就扔掉还能走的路
- 按「到下一密航点」把直段刹成爬行

旧 `CorridorFollower` 单测可留。

---

## 最近日志（回归，不是算法说明书）

| 日志 | 现象 | 处理 |
| --- | --- | --- |
| `shelf/delivery_20260906_07*` | `(1.066, 2.486)` 持续 hold | 禁区守卫 → 事件 + 退出，不能零输出 |
| `delivery_20260905_065322` | 两箱左右对拧 | 一图记两箱，整段重搜，安全层不选边 |
| `delivery_20260905_075736` | 西侧箱群 + 两点 fallback | 禁止未验证黄线 |
| `delivery_20260905_103938` | 黄线南切后撞箱 | 只执行可刹停前段 |
| `delivery_20260906_090209` | 黄线西转成功后假碰撞冻住 | 车心进实体墙才硬停；隔墙北沿回波不往黄线膨胀 |
| `delivery_20260906_091014` / `091258` | 箱区原地对切后回头 | `ROUTE_HOLD_S` + 拒绝大回头 |
| `delivery_20260906_092050` | A* 失败清路线死停 | 有未走路且未顶箱则 `keep_branch` |
| `delivery_20260906_092828` | 西侧箱缝 `safety_stop` 120 s | 顶箱允许掉头；雷达清楚时不要按未知地图爬行 |
| `delivery_20260906_093534` | 进配送区但 `arrived=false` | 接近点还差 16 cm + 超时；密航点刹车 |
| `delivery_20260906_094432` | **`arrived=true`，4.81 s** | 从残留位补完接近点；直段按剩余路径给速 |
