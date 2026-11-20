"""NumPy 数值计算练习：8 道基础题 + 4 道中等题。

运行方式：
    python numpy_homework.py

约定：
* 只依赖 Python 与 NumPy；
* 计算结果保留完整精度，打印时保留 6 位小数；
* 每题独立成段，函数题保留题目要求的函数名；
* 注释中给出输入/输出 shape 与关键表达式说明。
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)


def section(title):
    """打印统一的分节标题，便于对照 PDF 题目编号。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# NP-E1 数组、形状与切片
# ---------------------------------------------------------------------------
def exercise_e1():
    section("NP-E1 数组、形状与切片")
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])

    # 1. ndim 是轴数，shape 是各轴长度，size 是元素总数。
    print("A.ndim  =", A.ndim)
    print("A.shape =", A.shape)
    print("A.size  =", A.size)

    # 2. 下标从 0 开始：第 2 行是 A[1]；最后 2 列用负下标切片。
    row1 = A[1]
    last_two_cols = A[:, -2:]
    print("第 2 行 A[1]      =", row1, "shape =", row1.shape)
    print("最后 2 列 A[:, -2:] =", last_two_cols, "shape =", last_two_cols.shape)

    # 3. copy() 断开与 A 的内存共享，再原地把负数改成 0。
    B = A.copy()
    B[B < 0] = 0
    print("B = A.copy(); B[B < 0] = 0:\n", B)
    print("A 未被修改:\n", A)


# ---------------------------------------------------------------------------
# NP-E2 逐元素乘法、点乘与长度
# ---------------------------------------------------------------------------
def exercise_e2():
    section("NP-E2 逐元素乘法、点乘与长度")
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # 1. * 是逐元素乘法，结果仍是向量；@ 是点乘，结果是标量。
    elementwise = a * b
    dot = a @ b
    print("a * b =", elementwise, "shape =", elementwise.shape)
    print("a @ b =", dot, "shape =", np.shape(dot))

    # 2. 手写欧氏长度：平方 -> 求和 -> 开平方。
    norm_a = np.sqrt((a * a).sum())
    norm_b = np.sqrt((b * b).sum())
    print("||a|| =", norm_a, "  ||b|| =", norm_b)

    # 3. 用库函数核对。
    print("np.linalg.norm(a) =", np.linalg.norm(a))
    print("np.linalg.norm(b) =", np.linalg.norm(b))
    print("手写结果与库函数一致:",
          np.allclose([norm_a, norm_b], [np.linalg.norm(a), np.linalg.norm(b)]))


# ---------------------------------------------------------------------------
# NP-E3 两个向量的夹角
# ---------------------------------------------------------------------------
def angle_deg(a, b):
    """返回两个同维非零一维向量的夹角（度）。

    a, b: shape (D,) -> 标量。
    cos(theta) = (a @ b) / (||a|| * ||b||)，先 clip 到 [-1,1] 再 arccos。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("angle_deg 只接受一维向量")
    if a.shape != b.shape:
        raise ValueError("两个向量必须同维")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量没有定义夹角")

    cosine = (a @ b) / (norm_a * norm_b)
    # 浮点误差可能让余弦略超出 [-1,1]，clip 后再 arccos 才不会得到 nan。
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def exercise_e3():
    section("NP-E3 两个向量的夹角")
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    print("angle_deg(a, b) =", angle_deg(a, b), "度")
    print("预期 45 度，因为 cos(theta)=1/sqrt(2)")

    # 边界输入：零向量应抛出 ValueError。
    try:
        angle_deg(np.array([0.0, 0.0]), b)
    except ValueError as exc:
        print("零向量正确抛出 ValueError:", exc)


# ---------------------------------------------------------------------------
# NP-E4 两个点的距离
# ---------------------------------------------------------------------------
def exercise_e4():
    section("NP-E4 两个点的距离")
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    diff = p - q
    print("p - q =", diff, "shape =", diff.shape)
    # 欧氏距离：差向量平方和再开方。
    euclidean = np.sqrt((diff * diff).sum())
    # 曼哈顿距离：各坐标差绝对值之和。
    manhattan = np.abs(diff).sum()
    print("欧氏距离   =", euclidean)
    print("曼哈顿距离 =", manhattan)


# ---------------------------------------------------------------------------
# NP-E5 把向量变为单位长度
# ---------------------------------------------------------------------------
def unit_vector(a):
    """返回与 a 方向相同、长度为 1 的新数组；不修改原数组。

    a: shape (D,) -> shape (D,)。
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("unit_vector 只接受一维向量")
    length = np.linalg.norm(a)
    if length == 0:
        raise ValueError("零向量无法归一化")
    # 用 / 产生新数组，a 本身保持不变。
    return a / length


def exercise_e5():
    section("NP-E5 把向量变为单位长度")
    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print("a       =", a)
    print("单位向量 =", u)
    print("原数组未被修改:", np.array_equal(a, np.array([-1.20, 0.40, 0.80])))
    print("长度 =", np.linalg.norm(u))
    print("np.allclose(||u||, 1) =", np.allclose(np.linalg.norm(u), 1.0))

    try:
        unit_vector(np.array([0.0, 0.0, 0.0]))
    except ValueError as exc:
        print("零向量正确抛出 ValueError:", exc)


# ---------------------------------------------------------------------------
# NP-E6 沿不同 axis 汇总
# ---------------------------------------------------------------------------
def exercise_e6():
    section("NP-E6 沿不同 axis 汇总")
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])

    # axis=0 沿行方向压缩，得到每一列的和，shape 为 (3,)。
    col_sum = A.sum(axis=0)
    # axis=1 沿列方向压缩，得到每一行的均值，shape 为 (2,)。
    row_mean = A.mean(axis=1)
    print("A.sum(axis=0)  =", col_sum, "shape =", col_sum.shape)
    print("A.mean(axis=1) =", row_mean, "shape =", row_mean.shape)
    print("A.max()        =", A.max(), "shape =", np.shape(A.max()))
    print("解释：sum(axis=0)[j] 汇总第 j 列的两个元素；"
          "mean(axis=1)[i] 汇总第 i 行的三个元素。")


# ---------------------------------------------------------------------------
# NP-E7 一次广播，处理所有行
# ---------------------------------------------------------------------------
def exercise_e7():
    section("NP-E7 一次广播，处理所有行")
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    # X 是 (2,3)，scale/bias 是 (3,)，NumPy 把后者广播到每一行。
    Y = X * scale + bias
    print("Y =\n", Y)
    print("Y.shape =", Y.shape)
    print("解释：scale[j]、bias[j] 都作用在 X 的第 j 列，即两行共用同一组参数。")
    print("手算核对 Y[0,0] =", X[0, 0] * scale[0] + bias[0], "==", Y[0, 0])


# ---------------------------------------------------------------------------
# NP-E8 把两张图像组成一个 batch
# ---------------------------------------------------------------------------
def exercise_e8():
    section("NP-E8 把两张图像组成一个 batch")
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10

    # stack 新增第 0 轴：a、b 各是 (3,2,2)，batch 为 (2,3,2,2)。
    batch = np.stack([a, b], axis=0)
    print("batch.shape =", batch.shape, "（B,C,H,W）")
    # 蓝色通道是 C=2，取所有图像的第 2 个通道。
    blue = batch[:, 2]
    print("蓝色通道 batch[:, 2].shape =", blue.shape)
    # (B,C,H,W) -> (B,H,W,C)：把通道轴移到最后。
    hwc = batch.transpose(0, 2, 3, 1)
    print("转为 (B,H,W,C) 后 shape =", hwc.shape)


# ---------------------------------------------------------------------------
# NP-M1 批量计算夹角
# ---------------------------------------------------------------------------
def angles_to(query, X):
    """返回 X 的每一行与 query 的夹角（度），shape (N,)。

    query: (D,)；X: (N,D) -> (N,)。
    零向量行返回 np.nan；query 为零向量时抛出 ValueError。
    """
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)
    if query.ndim != 1 or X.ndim != 2:
        raise ValueError("query 必须是一维，X 必须是二维")
    if X.shape[1] != query.shape[0]:
        raise ValueError("query 与 X 的列数必须一致")

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("query 是零向量")

    # 按行求长度，keepdims 让结果保持 (N,1)，便于广播做除法。
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    # 零向量行的长度为 0；先标记，避免 0/0 警告。
    zero_rows = row_norms[:, 0] == 0
    # 复制一份长度并把零行改成 1，只用于安全除法；原长度仍用于判断。
    safe_norms = row_norms.copy()
    safe_norms[zero_rows] = 1.0
    # 一次矩阵乘法得到所有点积，再按行归一化。
    cosine = (X @ query) / (safe_norms[:, 0] * query_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    # 零向量行没有方向，夹角无定义，统一返回 nan。
    angles[zero_rows] = np.nan
    return angles


def exercise_m1():
    section("NP-M1 批量计算夹角")
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])
    angles = angles_to(query, X)
    print("angles_to(query, X) =", angles)
    print("shape =", angles.shape)
    print("说明：第 0、1 行与 query 同向/反向，第 3 行是零向量，返回 nan。")


# ---------------------------------------------------------------------------
# NP-M2 用余弦相似度做 Top-k 检索
# ---------------------------------------------------------------------------
def top_k_cosine(query, gallery, k):
    """返回 (选中行索引, 相似度)，按相似度从高到低，稳定排序。

    query: (D,)；gallery: (N,D)；k 为正整数。
    零向量行不参与排序；零 query、非法 k 抛出 ValueError。
    """
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if query.ndim != 1 or gallery.ndim != 2:
        raise ValueError("query 必须是一维，gallery 必须是二维")
    if gallery.shape[1] != query.shape[0]:
        raise ValueError("query 与 gallery 的列数必须一致")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError("k 必须是正整数")
    if k <= 0:
        raise ValueError("k 必须为正整数")

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("query 是零向量")

    row_norms = np.linalg.norm(gallery, axis=1)
    valid = row_norms != 0            # 零向量行不参与排序
    if k > valid.sum():
        raise ValueError("k 不能超过非零行数")

    valid_indices = np.flatnonzero(valid)
    # 只对有效行计算余弦相似度。
    cosine = (gallery[valid] @ query) / (row_norms[valid] * query_norm)
    cosine = np.clip(cosine, -1.0, 1.0)

    # 对负相似度做稳定升序排序，等价于对相似度做稳定降序，平分时保持原顺序。
    order = np.argsort(-cosine, kind="stable")[:k]
    return valid_indices[order], cosine[order]


def exercise_m2():
    section("NP-M2 用余弦相似度做 Top-k 检索")
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
    indices, scores = top_k_cosine(query, gallery, 3)
    print("选中行索引 =", indices)
    print("相似度     =", scores)
    print("选中 ID    =", ids[indices])
    print("img_zero 是零向量，已被排除在排序之外。")


# ---------------------------------------------------------------------------
# NP-M3 所有点对之间的距离矩阵
# ---------------------------------------------------------------------------
def exercise_m3():
    section("NP-M3 所有点对之间的距离矩阵")
    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])

    # A[:, None, :] 是 (3,1,2)，B[None, :, :] 是 (1,2,2)，广播成 (3,2,2)。
    diff = A[:, None, :] - B[None, :, :]
    print("差数组 shape =", diff.shape, "（A 的行, B 的行, 坐标）")
    D = np.sqrt((diff * diff).sum(axis=-1))
    print("D =\n", D)
    print("D.shape =", D.shape)
    nearest = np.argmin(D, axis=1)
    print("A 的每一行最近的 B 行索引 =", nearest)


# ---------------------------------------------------------------------------
# NP-M4 用矩阵运算完成一次前向计算
# ---------------------------------------------------------------------------
def exercise_m4():
    section("NP-M4 用矩阵运算完成一次前向计算")
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])

    # W1 是 (输出特征=2, 输入特征=3)，所以 X(2,3) @ W1.T(3,2) -> (2,2)。
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)
    # W2 是 (输出特征=2, 输入特征=2)，H(2,2) @ W2.T(2,2) -> (2,2)。
    Y = H @ W2.T + b2
    print("Z =\n", Z, "\nZ.shape =", Z.shape)
    print("H =\n", H, "\nH.shape =", H.shape)
    print("Y =\n", Y, "\nY.shape =", Y.shape)


def main():
    exercise_e1()
    exercise_e2()
    exercise_e3()
    exercise_e4()
    exercise_e5()
    exercise_e6()
    exercise_e7()
    exercise_e8()
    exercise_m1()
    exercise_m2()
    exercise_m3()
    exercise_m4()


if __name__ == "__main__":
    main()
