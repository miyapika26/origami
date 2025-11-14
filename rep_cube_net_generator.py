import math
import networkx as nx
from graphillion import GraphSet
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import argparse
import os

# --- 1. 面分割・隣接リスト自動生成 ---
def generate_cube_faces(k):
    n = int(math.isqrt(k))
    faces = list(range(6))
    squares = []
    for face in faces:
        for i in range(n):
            for j in range(n):
                squares.append((face, i, j))
    return faces, squares, n

# --- 2. 面の端の対応表（標準的な立方体の展開図に基づく） ---
# 面ID: 0=上, 1=下, 2=前, 3=後, 4=左, 5=右
# 各面の上下左右の隣接面と、どの端が対応するか
# (隣接面ID, 転送関数: (i,j,n)→(i',j'))
face_adjacency = {
    0: {'U': (3, lambda i, j, n: (0, n-1-j)),
        'D': (2, lambda i, j, n: (0, j)),
        'L': (4, lambda i, j, n: (j, 0)),
        'R': (5, lambda i, j, n: (n-1-j, 0))},
    1: {'U': (2, lambda i, j, n: (n-1, j)),
        'D': (3, lambda i, j, n: (n-1, n-1-j)),
        'L': (4, lambda i, j, n: (n-1-j, n-1)),
        'R': (5, lambda i, j, n: (j, n-1))},
    2: {'U': (0, lambda i, j, n: (n-1, j)),
        'D': (1, lambda i, j, n: (0, j)),
        'L': (4, lambda i, j, n: (i, n-1)),
        'R': (5, lambda i, j, n: (i, 0))},
    3: {'U': (0, lambda i, j, n: (0, n-1-j)),
        'D': (1, lambda i, j, n: (n-1, n-1-j)),
        'L': (5, lambda i, j, n: (n-1-i, n-1)),
        'R': (4, lambda i, j, n: (n-1-i, 0))},
    4: {'U': (0, lambda i, j, n: (j, 0)),
        'D': (1, lambda i, j, n: (n-1-j, 0)),
        'L': (3, lambda i, j, n: (n-1-i, 0)),
        'R': (2, lambda i, j, n: (i, 0))},
    5: {'U': (0, lambda i, j, n: (n-1-j, n-1)),
        'D': (1, lambda i, j, n: (j, n-1)),
        'L': (2, lambda i, j, n: (i, n-1)),
        'R': (3, lambda i, j, n: (n-1-i, n-1))}
}

def in_face_neighbors(face, i, j, n):
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < n and 0 <= nj < n:
            neighbors.append((face, ni, nj))
    return neighbors

def cross_face_neighbors(face, i, j, n):
    neighbors = []
    # 上端
    if i == 0 and 'U' in face_adjacency[face]:
        f2, trans = face_adjacency[face]['U']
        ni, nj = trans(i, j, n)
        neighbors.append((f2, ni, nj))
    # 下端
    if i == n-1 and 'D' in face_adjacency[face]:
        f2, trans = face_adjacency[face]['D']
        ni, nj = trans(i, j, n)
        neighbors.append((f2, ni, nj))
    # 左端
    if j == 0 and 'L' in face_adjacency[face]:
        f2, trans = face_adjacency[face]['L']
        ni, nj = trans(i, j, n)
        neighbors.append((f2, ni, nj))
    # 右端
    if j == n-1 and 'R' in face_adjacency[face]:
        f2, trans = face_adjacency[face]['R']
        ni, nj = trans(i, j, n)
        neighbors.append((f2, ni, nj))
    return neighbors

def build_adjacency(k):
    faces, squares, n = generate_cube_faces(k)
    adj = {}
    for face in faces:
        for i in range(n):
            for j in range(n):
                key = (face, i, j)
                neighbors = in_face_neighbors(face, i, j, n)
                neighbors += cross_face_neighbors(face, i, j, n)
                adj[key] = neighbors
    return adj, squares, n

# --- 3. グラフ構築と全域木列挙（Graphillion） ---
def build_graphillion_edges(adj):
    # Graphillion用のエッジはタプルの順序を揃える
    edges = set()
    for u, nbs in adj.items():
        for v in nbs:
            edge = tuple(sorted([u, v]))
            edges.add(edge)
    return list(edges)

def enumerate_spanning_trees(edges, squares):
    GraphSet.set_universe(edges)
    trees = GraphSet.trees()
    return trees

# --- 4. 展開図を2Dマトリックス上に展開（DFS） ---
def unfold_net(tree_edges, root, n):
    # 2Dマトリックス（十分大きいサイズ）
    size = n * 4  # 余裕を持たせる
    mat = np.zeros((size, size), dtype=int)
    visited = set()
    pos = {}
    # 面ごとの配置座標
    # 方向: 上, 下, 左, 右
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    # tree_edgesを隣接リストに
    tree_adj = {}
    for u, v in tree_edges:
        tree_adj.setdefault(u, []).append(v)
        tree_adj.setdefault(v, []).append(u)
    def dfs(node, x, y):
        visited.add(node)
        mat[x, y] = 1  # 1で埋める
        pos[node] = (x, y)
        for nb in tree_adj.get(node, []):
            if nb not in visited:
                # まだ配置していない隣接ノードを4方向のどこかに置く
                for dx, dy in dirs:
                    nx_, ny_ = x+dx, y+dy
                    if 0 <= nx_ < size and 0 <= ny_ < size and mat[nx_, ny_] == 0:
                        dfs(nb, nx_, ny_)
                        break
    dfs(root, size//2, size//2)
    return mat

# --- 5. 0/1行列化・重なり判定 ---
def is_valid_net(bin_mat, k):
    return np.sum(bin_mat) == 6*k

# --- 6. 正規化（回転・反転） ---
def canonical_form(bin_mat):
    forms = []
    for k in range(4):
        rot = np.rot90(bin_mat, k)
        forms.append(rot)
        forms.append(np.fliplr(rot))
    min_form = min(tuple(f.flatten()) for f in forms)
    return min_form

# --- 7. 非同型な展開図の抽出 ---
def extract_unique_nets(trees, n, k):
    unique_nets = set()
    count = 0
    print(f"展開図候補の処理を開始...")
    for tree in trees:
        count += 1
        # 100個ごとに進捗を表示
        if count % 100 == 0:
            print(f"  処理済み: {count}個, 抽出済み: {len(unique_nets)}個")
        
        # treeはエッジ集合
        # ルートを決める（最初のノード）
        root = tree[0][0]
        mat = unfold_net(tree, root, n)
        # 0でない部分だけを切り出し
        nonzero = np.argwhere(mat)
        if nonzero.size == 0:
            continue
        minx, miny = nonzero.min(axis=0)
        maxx, maxy = nonzero.max(axis=0)
        crop = mat[minx:maxx+1, miny:maxy+1]
        if not isinstance(crop, np.ndarray):
            crop = np.array(crop, dtype=int)
        else:
            crop = crop.astype(int)
        bin_mat = (crop > 0).astype(int)
        if is_valid_net(bin_mat, k):
            cf = canonical_form(bin_mat)
            unique_nets.add(cf)
    
    print(f"展開図候補処理完了: {count}個処理, {len(unique_nets)}個抽出")
    return unique_nets

# --- 展開図の可視化・保存・分割判定 ---
def plot_net(bin_mat, title=None):
    plt.figure(figsize=(4,4))
    plt.imshow(bin_mat, cmap='Greys', interpolation='none')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def save_net(bin_mat, filename):
    plt.figure(figsize=(4,4))
    plt.imshow(bin_mat, cmap='Greys', interpolation='none')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def can_be_split(bin_mat, k):
    # 必ずint型2次元配列に変換
    bin_mat = np.array(bin_mat, dtype=int)
    if bin_mat.ndim == 1:
        side = int(np.sqrt(bin_mat.size))
        bin_mat = bin_mat.reshape(side, side)
    structure = np.ones((3,3), dtype=int)
    labeled, num = scipy.ndimage.label(bin_mat, structure=structure)
    return num == k

def get_all_unit_cube_nets():
    # k=1の全ネット（canonical formのset）を事前に列挙
    k1_adj, k1_squares, k1_n = build_adjacency(1)
    k1_edges = build_graphillion_edges(k1_adj)
    k1_trees = enumerate_spanning_trees(k1_edges, k1_squares)
    k1_nets = extract_unique_nets(k1_trees, k1_n, 1)
    return k1_nets

def is_unit_cube_net(bin_mat, k1_nets):
    # 6マスで連結、かつk=1ネットと同型ならTrue
    if np.sum(bin_mat) != 6:
        return False
    cf = canonical_form(bin_mat)
    return cf in k1_nets

def save_colored_net(bin_mat, label_mat, filename):
    # 0は白、それ以外はラベルごとに色分け
    from matplotlib import colors
    cmap = plt.get_cmap('tab10')
    norm = colors.BoundaryNorm(boundaries=range(label_mat.max()+2), ncolors=label_mat.max()+1)
    plt.figure(figsize=(4,4))
    plt.imshow(label_mat, cmap=cmap, norm=norm, interpolation='none')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="レプ・キューブ展開図生成・判定")
    parser.add_argument('-k', type=int, default=4, help='分割数k（例: 4, 9, 16）')
    parser.add_argument('--max', type=int, default=5, help='最大出力数')
    parser.add_argument('--outdir', type=str, default='nets', help='画像保存ディレクトリ')
    parser.add_argument('--stop-on-first', action='store_true', help='レプ・キューブを一つ見つけたら停止')
    args = parser.parse_args()

    k = args.k
    max_out = args.max
    outdir = args.outdir
    stop_on_first = args.stop_on_first
    os.makedirs(outdir, exist_ok=True)

    print(f"k={k} で隣接リスト構築中...")
    adj, squares, n = build_adjacency(k)
    print(f"エッジ構築中...")
    edges = build_graphillion_edges(adj)
    print(f"全域木列挙中...（時間がかかる場合があります）")
    trees = enumerate_spanning_trees(edges, squares)
    print(f"全域木列挙完了。展開図候補数: {len(trees)}（GraphSetのlen()で数えられる場合のみ）")
    print(f"非同型な展開図抽出中...")
    unique_nets = extract_unique_nets(trees, n, k)
    print(f"非同型な展開図の数: {len(unique_nets)}")
    k1_nets = get_all_unit_cube_nets()
    
    rep_cube_found = False
    processed_count = 0
    for i, cf in enumerate(unique_nets):
        processed_count += 1
        print(f"--- 展開図 {i+1}/{len(unique_nets)} を処理中 ---")
        total = len(cf)
        h = int(np.sqrt(total))
        while total % h != 0:
            h -= 1
        w = total // h
        arr = np.array(cf, dtype=int).reshape(h, w)
        structure = np.ones((3,3), dtype=int)
        labeled, num = scipy.ndimage.label(arr, structure=structure)
        print(f"展開図 {i+1}: {num}成分")
        all_cube = True
        for l in range(1, num+1):
            part = (labeled==l).astype(int)
            if is_unit_cube_net(part, k1_nets):
                print(f"  成分{l}: 立方体ネットOK")
            else:
                print(f"  成分{l}: 立方体ネットNG")
                all_cube = False
        print(f"展開図 {i+1} の配列:\n{arr}")
        print(f"ラベル配列:\n{labeled}")
        print(f"保存ファイル: {os.path.join(outdir, f'colored_net_{i+1}.png')}")
        save_colored_net(arr, labeled, os.path.join(outdir, f'colored_net_{i+1}.png'))
        if all_cube and num == k:
            print(f'🎉 レプ・キューブ発見！ 🎉')
            print(f'この展開図は「レプ・キューブ」条件を満たします')
            print(f'展開図番号: {i+1}')
            print(f'成分数: {num}')
            print(f'配列サイズ: {h}×{w}')
            rep_cube_found = True
            if stop_on_first:
                print(f'--stop-on-first オプションにより処理を停止します')
                break
        else:
            print(f'この展開図は「レプ・キューブ」条件を満たしません')
        
        # 10個ごとに処理済み数を表示
        if processed_count % 10 == 0:
            print(f"📊 処理済み展開図: {processed_count}/{len(unique_nets)}個")
        
        if i >= max_out-1:
            break
    
    if rep_cube_found:
        print(f'\n✅ レプ・キューブが見つかりました！')
    else:
        print(f'\n❌ レプ・キューブは見つかりませんでした。')
        print(f'（処理した展開図数: {processed_count}）')

if __name__ == "__main__":
    main() 