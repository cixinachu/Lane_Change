import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from NGSIM_env.envs.env import NGSIMEnv
from NGSIM_env.road.lane import StraightLane


def _lane_polygon(lane: StraightLane):
    """生成直线路段的填充多边形，用于更平滑的视觉效果。"""
    x0, y0 = lane.start
    x1, y1 = lane.end
    dx, dy = x1 - x0, y1 - y0
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return None
    nx, ny = -dy / length, dx / length  # 单位法向
    hw = lane.width / 2
    return [
        (x0 + nx * hw, y0 + ny * hw),
        (x1 + nx * hw, y1 + ny * hw),
        (x1 - nx * hw, y1 - ny * hw),
        (x0 - nx * hw, y0 - ny * hw),
    ]


def visualize_road_network(scene_name='us-101'):
    env = NGSIMEnv(scene=scene_name)
    env.reset()
    network = env.road.network

    plt.figure(figsize=(16, 6), facecolor="#f8fafc")
    ax = plt.gca()
    ax.set_facecolor("#eef2f6")

    # 车道填充与中心线
    for lane in network.lanes_list():
        poly_pts = _lane_polygon(lane)
        if poly_pts is None:
            continue
        lane_patch = Polygon(poly_pts, closed=True, facecolor="#d7e3f4", edgecolor="#ffffff", linewidth=1.2, alpha=0.95)
        ax.add_patch(lane_patch)
        x0, y0 = lane.start
        x1, y1 = lane.end
        ax.plot([x0, x1], [y0, y1], color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)

    # 车辆（高对比色块）
    def draw_vehicle(label, veh_dict, length, width, color):
        cx = veh_dict['y']  # 数据中 y 为纵向，x 为横向，需对调
        cy = veh_dict['x']
        rect = Rectangle(
            (cx - length / 2, cy - width / 2),
            length,
            width,
            facecolor=color,
            edgecolor="#111827",
            linewidth=1.0,
            alpha=0.9,
            label=label
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, color="#ffffff", fontsize=9, ha='center', va='center', weight='bold')

    draw_vehicle('LCV', env.lcv_veh, env.lcv_length, env.lcv_width, "#ef4444")
    draw_vehicle('FV', env.fv_veh, env.fv_length, env.fv_width, "#3b82f6")
    draw_vehicle('NLV', env.nlv_veh, env.nlv_length, env.nlv_width, "#f59e0b")
    draw_vehicle('OLV', env.olv_veh, env.olv_length, env.olv_width, "#10b981")

    # 去掉坐标轴与网格，保持科研海报风
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"NGSIM Road Snapshot · {scene_name}", fontsize=16, weight='bold', color="#0f172a")
    ax.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_road_network('us-101')
