# 菱形三十面体の展開図の作成プログラム
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def add_golden_rhombus(ax, center, angle_deg=0, color_value=0, cmap=None, norm=None):
    """
    指定されたmatplotlibのAxesに、中心座標と角度を指定して
    黄金菱形 (Golden Rhombus) を1つ追加する関数。
    color_valueに数値を指定する。
    """
    phi = (1 + np.sqrt(5)) / 2
    base_vertices = np.array([
        [phi, 0], [0, 1], [-phi, 0], [0, -1]
    ])
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated_vertices = np.dot(base_vertices, rotation_matrix.T)
    translated_vertices = rotated_vertices + center

    # STEP 2: 数値を色(RGBA)に変換
    final_color = cmap(norm(color_value))
    
    rhombus = Polygon(translated_vertices, facecolor=final_color, edgecolor='black', linewidth=1)
    ax.add_patch(rhombus)
    return translated_vertices

 
def draw_row():
    fig, ax = plt.subplots()
    angle_deg = 0
    center = (0, 0)
    f_center = center # 最初の中心座標を保存
    # STEP 1: 色の物差しとパレットを準備
    # 色として使う数値の範囲を0から10に設定
    norm = mcolors.Normalize(vmin=0, vmax=10)
    # 使用するカラーマップを'viridis'に設定 (他の例: 'jet', 'coolwarm', 'magma')
    cmap = cm.get_cmap('viridis')
    phi = (1 + np.sqrt(5)) / 2
    color_value = 0
    for k in range(3):
        for i in range(2):
            for j in range(5):
                add_golden_rhombus(ax,center,angle_deg,color_value,cmap, norm)
                center=(center[0]+ 2 * phi , center[1])
            center=(f_center[0] + phi , f_center[1] - 1)
            f_center = center
            color_value += 2
        center = (0, f_center[1])
        f_center = center
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    ax.set_title("Rhombic Triacontahedron Net")
    ax.axis('off')
    
    plt.show()
# --- メイン処理 ---
if __name__ == "__main__":

    draw_row()
