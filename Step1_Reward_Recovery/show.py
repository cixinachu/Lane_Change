import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

# 手动选择要可视化的场景索引（0-based）。修改这个数字即可切换场景。
SCENE_IDX = 3


def plot_scene(case, scene_idx=0):
    # 多车字段映射：车辆名称 -> (y列, x列)
    vehicles = {
        'LCV': ('y_x', 'x_x'),
        'FV':  ('y_y', 'x_y'),
        'NLV': ('y',   'x'),
        'OLV': ('y_z', 'x_z'),
    }

    dt = 0.1  # 数据时间间隔

    # 收集所有轨迹的合并范围（使用原始单位英尺，刻度遵循 pkl）
    all_y = pd.concat([case[v[0]] for v in vehicles.values()])  # longitudinal
    all_x = pd.concat([case[v[1]] for v in vehicles.values()])  # lateral
    y_min, y_max = all_y.min(), all_y.max()
    x_min, x_max = all_x.min(), all_x.max()

    plt.figure(figsize=(8, 10), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # 计算全局速度范围，用于统一 colorbar
    global_speeds = []
    traj_data = {}
    for name, (y_col, x_col) in vehicles.items():
        df = case.loc[:, [y_col, x_col]].copy()
        df['x_raw'] = df[x_col]          # lateral ft
        df['y_raw'] = df[y_col]          # longitudinal ft
        # 优先使用 pkl 中记录的速度字段（如果存在），否则用差分估算
        suffix = x_col[1:] if x_col.startswith('x') else ''  # 提取后缀，如 '_x', '_y', '_z' 或 ''
        vx_col = f"v_x{suffix}"
        vy_col = f"v_y{suffix}"
        if vx_col in case.columns and vy_col in case.columns:
            vx = case[vx_col] * 0.3048  # ft/s -> m/s
            vy = case[vy_col] * 0.3048
            speed = np.sqrt(vx ** 2 + vy ** 2)
            print(f"Using recorded speed for {name} from columns {vx_col}, {vy_col}.")
        else:
            dx = df['x_raw'].diff().fillna(0)
            dy = df['y_raw'].diff().fillna(0)
            speed = np.sqrt(dx**2 + dy**2) / dt * 0.3048  # 差分估计 (m/s)
            print(f"Estimated speed for {name} using finite difference.")

        df['speed_mps'] = speed
        traj_data[name] = df
        global_speeds.append(speed)

    if global_speeds:
        speeds_concat = pd.concat(global_speeds)
        vmin, vmax = speeds_concat.min(), speeds_concat.max()
    else:
        vmin, vmax = 0, 1

    # 仅用线型/点型区分车辆，颜色统一
    styles = {
        'LCV': {'ls': '-'},
        'FV':  {'ls': '--'},
        'NLV': {'ls': '-.'},
        'OLV': {'ls': ':'}
    }
    markers = {
        'LCV': 'o',
        'FV': 's',
        'NLV': '^',
        'OLV': 'x'
    }
    line_color = "#444444"

    # 绘制轨迹（横轴 longitudinal，纵轴 lateral）——整体交换坐标
    scatters = []
    for name, df in traj_data.items():
        sc = ax.scatter(df['y_raw'], df['x_raw'], c=df['speed_mps'],
                        cmap='jet', s=8, alpha=0.9, linewidths=0,
                        vmin=vmin, vmax=vmax, label=name)
        # 叠加线条轮廓以区分形状
        st = styles.get(name, {'ls': '-'})
        mk = markers.get(name, None)
        ax.plot(df['y_raw'], df['x_raw'], linestyle=st['ls'],
                linewidth=1.5, alpha=0.9, color=line_color,
                marker=mk, markevery=10, markersize=4,
                markerfacecolor='none', markeredgecolor=line_color,
                markeredgewidth=1.0)
        scatters.append(sc)

    # 参考虚线（横向均值）
    dash_y = all_y.mean()
    ax.axvline(dash_y, linestyle='--', color='k', linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Lateral (x, ft)", fontsize=12)
    ax.set_ylabel("Longitudinal (y, ft)", fontsize=12)
    ax.set_title(f"sample_data.pkl Scene {scene_idx}", fontsize=14, weight='bold')
    ax.set_xlim(y_min - 5, y_max + 5)
    ax.set_ylim(x_min - 5, x_max + 5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(False)
    handles = [Line2D([0], [0], color=line_color,
                      linestyle=styles[name]['ls'], linewidth=2.0,
                      marker=markers[name], markersize=6,
                      markerfacecolor='none', markeredgecolor=line_color,
                      markeredgewidth=1.0,
                      label=name)
               for name in styles.keys()]
    ax.legend(handles=handles, frameon=True, loc='lower right', fontsize=9,
              facecolor='white', edgecolor='#cbd5e1', framealpha=0.9)

    # 颜色条使用最后一个 scatter，范围统一 vmin/vmax
    cbar = plt.colorbar(scatters[-1], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed (m/s)", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data" / "sample_data_with_speed.pkl"
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    # 支持两种格式：原始 list（scene）或 dict(lcv/fv/nlv/olv)
    if isinstance(raw, dict):
        scenes = []
        for lcv, fv, nlv, olv in zip(raw['lcv'], raw['fv'], raw['nlv'], raw['olv']):
            pair = pd.DataFrame({
                'y_x': lcv['y'], 'x_x': lcv['x'],
                'y_y': fv['y'], 'x_y': fv['x'],
                'y':   nlv['y'], 'x':   nlv['x'],
                'y_z': olv['y'], 'x_z': olv['x'],
                'v_x':   nlv['v_x'], 'v_y':   nlv['v_y'],
                'v_x_x': lcv['v_x'], 'v_y_x': lcv['v_y'],
                'v_x_y': fv['v_x'],  'v_y_y': fv['v_y'],
                'v_x_z': olv['v_x'], 'v_y_z': olv['v_y'],
            })
            scenes.append(pair)
        data = scenes
    else:
        data = raw

    total = len(data)
    # 索引越界时自动回退到最后一个有效场景
    scene_idx = SCENE_IDX
    if scene_idx < 0 or scene_idx >= total:
        print(f"[Warn] SCENE_IDX={scene_idx} out of range [0, {total-1}], fallback to last scene.")
        scene_idx = total - 1

    print(f"Total scenes: {total} | using scene {scene_idx}")
    plot_scene(data[scene_idx], scene_idx=scene_idx)
