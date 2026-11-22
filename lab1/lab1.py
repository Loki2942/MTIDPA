import os
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances

from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/epogrebnyak/ru-cities/"
    "main/assets/towns.csv"
)
EARTH_RADIUS_KM = 6371.0

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
DATA_FILE = os.path.join(DATA_FOLDER, "towns.csv")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "clusters")

MAX_K = 20
RANDOM_STATE = 42

# Веса критериев при выборе числа кластеров в авторском методе.
# Сумма весов должна быть равна 1.
WEIGHT_SILHOUETTE = 0.65
WEIGHT_COMPACTNESS = 0.25
WEIGHT_BALANCE = 0.10


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def save_csv(df, path):
    """Сохранение таблицы в CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def finish_plot(path):
    """Завершение и сохранение текущего графика."""
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


def prepare_folder(method_name):
    """Создание чистой папки для результатов метода."""
    folder = os.path.join(RESULTS_FOLDER, method_name)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    return folder


def save_clusters(df, folder):
    """Сохранение общей таблицы и отдельных файлов кластеров."""
    save_csv(df, os.path.join(folder, "all_cities.csv"))

    for cluster, cluster_df in df.groupby("cluster", sort=True):
        save_csv(
            cluster_df,
            os.path.join(folder, f"cluster_{cluster}.csv")
        )

    print("\nРезультаты сохранены:", os.path.abspath(folder), sep="\n")


def finish_result_plot(path):
    """Сохранение графика без обязательной легенды."""
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


# ============================================================
# ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
# ============================================================

def load_data():
    print(
        "\n" + "=" * 70,
        "ЗАГРУЗКА БАЗЫ ГОРОДОВ РОССИИ",
        "=" * 70,
        sep="\n"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print("\nИспользуется локальная база данных:", DATA_FILE, sep="\n")

        try:
            df = pd.read_csv(DATA_FILE)
        except Exception as error:
            print("\nОшибка чтения базы:", error, sep="\n")
            return None
    else:
        print(
            "\nЛокальная база данных не найдена.",
            "Выполняется первоначальная загрузка...",
            sep="\n"
        )

        try:
            df = pd.read_csv(DATA_URL)
        except Exception as error:
            print(
                "\nНе удалось скачать базу.",
                "Проверьте подключение к интернету.",
                error,
                sep="\n"
            )
            return None

        try:
            save_csv(df, DATA_FILE)
            print(
                "\nБаза успешно скачана.",
                f"Файл: {DATA_FILE}",
                "При следующих запусках скачивание выполняться не будет.",
                sep="\n"
            )
        except Exception as error:
            print("\nОшибка сохранения базы:", error, sep="\n")
            return None

    required_columns = ["city", "lat", "lon"]

    for column in required_columns:
        if column not in df.columns:
            print(f"\nОшибка: отсутствует столбец '{column}'.")
            return None

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    if "population" in df.columns:
        df["population"] = pd.to_numeric(
            df["population"], errors="coerce"
        )

    df = df.dropna(subset=["city", "lat", "lon"]).copy()
    df = df[
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    ]
    df = df.reset_index(drop=True)

    print("\nБаза готова.")
    print(f"Количество городов: {len(df)}")

    if len(df) < 3:
        print("Для кластеризации необходимо не менее трёх городов.")
        return None

    return df


def show_data_info(df):
    print(
        "\n" + "=" * 70,
        "ИНФОРМАЦИЯ О ДАННЫХ",
        "=" * 70,
        f"\nКоличество городов: {len(df)}",
        "\nПоля базы:",
        sep="\n"
    )

    for column in df.columns:
        print(f" - {column}")

    possible_columns = [
        "city", "region_name", "federal_district",
        "population", "lat", "lon"
    ]
    columns = [column for column in possible_columns if column in df.columns]

    print(
        "\nПервые 20 городов:\n",
        df[columns].head(20).to_string(index=False),
        sep="\n"
    )


# ============================================================
# РАСЧЁТ РАССТОЯНИЙ
# ============================================================

def calculate_distance_matrix(df):
    """
    Расчёт матрицы географических расстояний между городами
    по формуле гаверсинусов.
    """
    coordinates = df[["lat", "lon"]].to_numpy(dtype=float)
    coordinates_rad = np.radians(coordinates)

    distance_matrix = (
        haversine_distances(coordinates_rad) * EARTH_RADIUS_KM
    )

    return distance_matrix


# ============================================================
# МЕТОД ЛОКТЯ ДЛЯ K-MEANS
# ============================================================

def find_elbow(x_values, y_values):
    """Поиск точки, наиболее удалённой от линии между концами графика."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if len(x) < 3:
        return 0

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        return 0

    points = np.column_stack((
        (x - x.min()) / x_range,
        (y - y.min()) / y_range
    ))

    first_point = points[0]
    last_point = points[-1]
    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return 0

    line_vector /= line_length
    vectors = points - first_point
    projections = np.outer(vectors @ line_vector, line_vector)
    distances = np.linalg.norm(vectors - projections, axis=1)

    return int(np.argmax(distances))


# ============================================================
# МЕТОД №1 — K-MEANS
# ============================================================

def run_kmeans(df):
    print(
        "\n" + "=" * 80,
        "МЕТОД №1 — K-MEANS + МЕТОД ЛОКТЯ",
        "=" * 80,
        sep="\n"
    )

    if len(df) < 3:
        print("\nНедостаточно городов.")
        return None

    start_time = time.perf_counter()

    # Координаты широты и долготы используются как признаки.
    X = df[["lat", "lon"]].to_numpy(dtype=float)

    max_k = min(MAX_K, len(df) - 1)
    k_values = list(range(1, max_k + 1))
    inertias = []
    models = []

    print("\nПеребор количества кластеров:")
    print(f"{'K':<10}{'WCSS':>20}")
    print("-" * 30)

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )
        model.fit(X)

        models.append(model)
        inertias.append(model.inertia_)
        print(f"{k:<10}{model.inertia_:>20.2f}")

    optimal_index = find_elbow(k_values, inertias)
    optimal_k = k_values[optimal_index]
    model = models[optimal_index]
    labels = model.labels_.copy()

    result = df.copy()
    result["cluster"] = labels + 1

    distance_matrix = calculate_distance_matrix(df)
    silhouette = None

    if 1 < optimal_k < len(df):
        try:
            silhouette = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed"
            )
        except ValueError:
            silhouette = None

    execution_time = time.perf_counter() - start_time

    print(f"\nОптимальное количество кластеров K = {optimal_k}")

    if silhouette is not None:
        print(f"Silhouette Score: {silhouette:.4f}")

    print(f"Время вычислений: {execution_time:.4f} с")

    for cluster in range(1, optimal_k + 1):
        cluster_df = result[result["cluster"] == cluster]
        print(
            f"\nКластер {cluster}: {len(cluster_df)} городов",
            ", ".join(cluster_df["city"].astype(str).tolist()),
            sep="\n"
        )

    folder = prepare_folder("kmeans")
    save_clusters(result, folder)

    save_csv(
        pd.DataFrame({"k": k_values, "wcss": inertias}),
        os.path.join(folder, "k_search.csv")
    )

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, inertias, marker="o")
    plt.scatter(
        optimal_k,
        inertias[optimal_index],
        s=150,
        label=f"Оптимальное K = {optimal_k}"
    )
    plt.xlabel("Количество кластеров K")
    plt.ylabel("WCSS")
    plt.title("K-means — определение K методом локтя")
    finish_plot(os.path.join(folder, "elbow.png"))

    plt.figure(figsize=(14, 8))
    plt.scatter(
        result["lon"], result["lat"],
        c=result["cluster"], cmap="tab20", s=25
    )
    centers = model.cluster_centers_
    plt.scatter(
        centers[:, 1], centers[:, 0],
        marker="X", s=200, c="black",
        label="Центры кластеров"
    )
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.title(f"K-means: K = {optimal_k}")
    finish_plot(os.path.join(folder, "clusters.png"))

    return {
        "method": "K-means",
        "clusters": optimal_k,
        "parameter": optimal_k,
        "parameter_name": "K",
        "silhouette": silhouette,
        "execution_time": execution_time,
        "result": result
    }


# ============================================================
# АВТОРСКИЙ МЕТОД — АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ
# ============================================================

def calculate_medoid(distance_matrix, member_indices):
    """
    Поиск медоида кластера.
    Медоид — реальный город, сумма расстояний от которого
    до остальных городов кластера минимальна.
    """
    cluster_distances = distance_matrix[
        np.ix_(member_indices, member_indices)
    ]
    sums = cluster_distances.sum(axis=1)

    return int(member_indices[np.argmin(sums)])


def calculate_cluster_compactness(distance_matrix, labels):
    """
    Оценка компактности кластеров в диапазоне [0, 1].

    Сначала рассчитывается среднее расстояние между парами городов
    внутри каждого кластера. Затем оно сравнивается со средним
    расстоянием между всеми парами городов набора данных.
    Чем меньше внутрикластерные расстояния, тем выше оценка.
    """
    n = len(labels)
    global_mean = distance_matrix[np.triu_indices(n, k=1)].mean()

    if global_mean <= 0:
        return 1.0

    weighted_sum = 0.0
    total_pairs = 0

    for cluster in np.unique(labels):
        indices = np.flatnonzero(labels == cluster)

        if len(indices) < 2:
            continue

        submatrix = distance_matrix[np.ix_(indices, indices)]
        pair_distances = submatrix[np.triu_indices(len(indices), k=1)]
        pair_count = len(pair_distances)

        weighted_sum += pair_distances.sum()
        total_pairs += pair_count

    # Если все кластеры одноэлементные, компактность не определена.
    if total_pairs == 0:
        return 0.0

    within_mean = weighted_sum / total_pairs
    return float(np.clip(1.0 - within_mean / global_mean, 0.0, 1.0))


def evaluate_cluster_counts(distance_matrix, hierarchy, max_k=MAX_K):
    """
    Проверяет несколько вариантов количества кластеров и выбирает
    лучший по взвешенной оценке из трёх критериев:
      1. Silhouette Score — разделённость кластеров;
      2. Compactness — географическая компактность;
      3. Balance — равномерность размеров кластеров.

    Возвращает выбранное K и таблицу оценок для всех кандидатов.
    """
    n = len(distance_matrix)
    upper_k = min(max_k, n - 1)

    if upper_k < 2:
        raise ValueError("Для оценки необходимо не менее трёх объектов.")

    rows = []

    for k in range(2, upper_k + 1):
        labels = cut_tree(hierarchy, n_clusters=[k]).flatten()

        try:
            silhouette = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed"
            )
        except ValueError:
            silhouette = -1.0

        # Переводим Silhouette из [-1, 1] в [0, 1].
        silhouette_normalized = float(np.clip((silhouette + 1.0) / 2.0, 0, 1))
        compactness = calculate_cluster_compactness(distance_matrix, labels)
        balance = calculate_cluster_balance(labels)

        total_score = (
            WEIGHT_SILHOUETTE * silhouette_normalized
            + WEIGHT_COMPACTNESS * compactness
            + WEIGHT_BALANCE * balance
        )

        rows.append({
            "k": k,
            "silhouette": silhouette,
            "silhouette_normalized": silhouette_normalized,
            "compactness": compactness,
            "balance": balance,
            "total_score": total_score
        })

    metrics = pd.DataFrame(rows)
    best_index = metrics["total_score"].idxmax()
    optimal_k = int(metrics.loc[best_index, "k"])

    return optimal_k, metrics

def calculate_cluster_balance(labels):
    """Оценка равномерности размеров кластеров от 0 до 1."""
    _, counts = np.unique(labels, return_counts=True)
    cluster_count = len(counts)

    if cluster_count <= 1:
        return 0.0

    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(cluster_count)

    return float(entropy / max_entropy)


def run_custom(df):
    print(
        "\n" + "=" * 80,
        "МЕТОД №2 — АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ (АОК)",
        "=" * 80,
        sep="\n"
    )

    if len(df) < 3:
        print("\nНедостаточно городов.")
        return None

    start_time = time.perf_counter()

    print("\nРасчёт матрицы географических расстояний...")
    distance_matrix = calculate_distance_matrix(df)
    print("Матрица расстояний рассчитана.")

    print("\nПостроение иерархии кластеров...")

    # linkage принимает condensed distance matrix.
    condensed_distances = squareform(
        distance_matrix,
        checks=False
    )

    # Average Linkage — среднее расстояние между всеми
    # парами объектов двух объединяемых кластеров.
    hierarchy = linkage(
        condensed_distances,
        method="average"
    )

    # Проверяем несколько вариантов K и выбираем лучший
    # по многокритериальной оценке.
    optimal_k, candidate_metrics = evaluate_cluster_counts(
        distance_matrix,
        hierarchy,
        max_k=MAX_K
    )

    print("\nОценка вариантов количества кластеров:")
    print(candidate_metrics.round(4).to_string(index=False))
    print(f"\nВыбрано K = {optimal_k} по максимальной общей оценке.")

    # Получение меток кластеров на выбранном уровне иерархии.
    labels = cut_tree(hierarchy, n_clusters=[optimal_k]).flatten()

    result = df.copy()
    result["cluster"] = labels + 1
    result["is_cluster_center"] = False
    result["distance_to_center_km"] = 0.0

    centers = []

    for cluster in range(optimal_k):
        member_indices = np.flatnonzero(labels == cluster)

        if len(member_indices) == 0:
            continue

        medoid_index = calculate_medoid(
            distance_matrix,
            member_indices
        )
        centers.append(medoid_index)

        result.loc[medoid_index, "is_cluster_center"] = True

        result.loc[
            member_indices, "distance_to_center_km"
        ] = distance_matrix[
            medoid_index, member_indices
        ]

    selected_metrics = candidate_metrics[
        candidate_metrics["k"] == optimal_k
    ].iloc[0]
    silhouette = float(selected_metrics["silhouette"])
    compactness = float(selected_metrics["compactness"])
    balance = float(selected_metrics["balance"])
    total_score = float(selected_metrics["total_score"])
    execution_time = time.perf_counter() - start_time

    print(
        "\n" + "=" * 80,
        "РЕЗУЛЬТАТ АГЛОМЕРАТИВНОЙ КЛАСТЕРИЗАЦИИ",
        "=" * 80,
        f"\nКоличество кластеров: {optimal_k}",
        sep="\n"
    )

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Compactness: {compactness:.4f}")
    print(f"Balance: {balance:.4f}")
    print(f"Общая оценка Q: {total_score:.4f}")
    print(f"Время вычислений: {execution_time:.4f} с")

    for cluster in range(1, optimal_k + 1):
        cluster_df = result[result["cluster"] == cluster]
        center_df = cluster_df[cluster_df["is_cluster_center"]]

        center_name = (
            center_df.iloc[0]["city"]
            if not center_df.empty else "—"
        )

        max_distance = cluster_df["distance_to_center_km"].max()

        print(
            f"\nКластер {cluster}: {len(cluster_df)} городов",
            f"Медоид: {center_name}",
            f"Максимальное расстояние до медоида: {max_distance:.1f} км",
            ", ".join(cluster_df["city"].astype(str).tolist()),
            sep="\n"
        )

    folder = prepare_folder("agglomerative")
    save_clusters(result, folder)

    # Сохранение истории объединений.
    merge_history = pd.DataFrame({
        "merge_number": np.arange(1, len(hierarchy) + 1),
        "distance_km": hierarchy[:, 2],
        "clusters_after_merge": (
            len(df) - np.arange(1, len(hierarchy) + 1)
        )
    })
    save_csv(
        merge_history,
        os.path.join(folder, "merge_history.csv")
    )

    # Сохранение оценок всех проверенных вариантов K.
    save_csv(
        candidate_metrics,
        os.path.join(folder, "candidate_metrics.csv")
    )

    # Сохранение параметров выбранного решения и весов критериев.
    selected_info = pd.DataFrame([{
        "optimal_clusters": optimal_k,
        "silhouette": silhouette,
        "compactness": compactness,
        "balance": balance,
        "total_score": total_score,
        "weight_silhouette": WEIGHT_SILHOUETTE,
        "weight_compactness": WEIGHT_COMPACTNESS,
        "weight_balance": WEIGHT_BALANCE
    }])
    save_csv(selected_info, os.path.join(folder, "optimal_solution.csv"))

    # График общей оценки для каждого проверенного K.
    plt.figure(figsize=(10, 6))
    plt.plot(
        candidate_metrics["k"],
        candidate_metrics["total_score"],
        marker="o",
        label="Общая оценка Q(K)"
    )
    plt.scatter(
        [optimal_k],
        [total_score],
        s=150,
        label=f"Выбрано K = {optimal_k}"
    )
    plt.xlabel("Количество кластеров K")
    plt.ylabel("Общая оценка качества")
    plt.title("АОК — многокритериальный выбор количества кластеров")
    finish_plot(os.path.join(folder, "quality_by_k.png"))

    # График расстояний объединения.
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(hierarchy) + 1),
        hierarchy[:, 2],
        linewidth=1
    )
    plt.axhline(
        y=hierarchy[-optimal_k, 2] if optimal_k < len(df) else hierarchy[0, 2],
        linestyle="--",
        label=f"Уровень выбранного разреза (K = {optimal_k})"
    )
    plt.xlabel("Номер объединения")
    plt.ylabel("Расстояние объединения, км")
    plt.title("АОК — изменение расстояний при объединении кластеров")
    finish_plot(os.path.join(folder, "merge_distances.png"))

    # Дендрограмма. При большом числе городов показываем
    # верхние уровни дерева, чтобы график оставался читаемым.
    plt.figure(figsize=(14, 7))
    dendrogram(
        hierarchy,
        truncate_mode="lastp",
        p=min(40, len(df)),
        show_leaf_counts=True,
        leaf_rotation=90
    )
    plt.xlabel("Группы городов")
    plt.ylabel("Расстояние объединения, км")
    plt.title("Дендрограмма агломеративной кластеризации")
    finish_result_plot(os.path.join(folder, "dendrogram.png"))

    # Географическая визуализация кластеров.
    plt.figure(figsize=(14, 8))
    plt.scatter(
        result["lon"], result["lat"],
        c=result["cluster"], cmap="tab20", s=25
    )

    center_rows = result[result["is_cluster_center"]]
    plt.scatter(
        center_rows["lon"], center_rows["lat"],
        marker="X", s=180, c="black",
        label="Медоиды кластеров"
    )
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.title(f"АОК: количество кластеров = {optimal_k}")
    finish_plot(os.path.join(folder, "clusters.png"))

    return {
        "method": "Agglomerative clustering",
        "clusters": optimal_k,
        "parameter": optimal_k,
        "parameter_name": "Количество кластеров",
        "silhouette": silhouette,
        "compactness": compactness,
        "balance": balance,
        "total_score": total_score,
        "execution_time": execution_time,
        "result": result
    }


# ============================================================
# СРАВНЕНИЕ МЕТОДОВ
# ============================================================

def compare_methods(kmeans_result, custom_result):
    if kmeans_result is None or custom_result is None:
        print("\nСравнение невозможно: один из методов не выполнен.")
        return

    k_silhouette = (
        f"{kmeans_result['silhouette']:.4f}"
        if kmeans_result["silhouette"] is not None else "—"
    )
    custom_silhouette = (
        f"{custom_result['silhouette']:.4f}"
        if custom_result["silhouette"] is not None else "—"
    )

    print(
        "\n" + "=" * 90,
        "СРАВНЕНИЕ МЕТОДОВ",
        "=" * 90,
        f"\n{'Показатель':<32}{'K-means':<25}{'АОК':<25}",
        "-" * 82,
        f"{'Количество кластеров':<32}"
        f"{kmeans_result['clusters']:<25}"
        f"{custom_result['clusters']:<25}",
        f"{'Silhouette Score':<32}"
        f"{k_silhouette:<25}"
        f"{custom_silhouette:<25}",
        f"{'Компактность':<32}{'—':<25}"
        f"{custom_result['compactness']:<25.4f}",
        f"{'Balance':<32}{'—':<25}"
        f"{custom_result['balance']:<25.4f}",
        f"{'Общая оценка АОК':<32}{'—':<25}"
        f"{custom_result['total_score']:<25.4f}",
        f"{'Время вычислений, с':<32}"
        f"{kmeans_result['execution_time']:<25.4f}"
        f"{custom_result['execution_time']:<25.4f}",
        sep="\n"
    )

    comparison = pd.DataFrame({
        "method": ["K-means", "Agglomerative clustering"],
        "clusters": [
            kmeans_result["clusters"],
            custom_result["clusters"]
        ],
        "silhouette": [
            kmeans_result["silhouette"],
            custom_result["silhouette"]
        ],
        "compactness": [
            np.nan,
            custom_result["compactness"]
        ],
        "balance": [
            np.nan,
            custom_result["balance"]
        ],
        "total_score": [
            np.nan,
            custom_result["total_score"]
        ],
        "execution_time_seconds": [
            kmeans_result["execution_time"],
            custom_result["execution_time"]
        ]
    })

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    comparison_file = os.path.join(RESULTS_FOLDER, "comparison.csv")
    save_csv(comparison, comparison_file)

    print("\nСравнение сохранено:", comparison_file, sep="\n")


# ============================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================

def main():
    print(
        "\n" + "=" * 70,
        "КЛАСТЕРИЗАЦИЯ ГОРОДОВ РОССИИ",
        "Сравнение K-means и авторского агломеративного метода",
        "=" * 70,
        sep="\n"
    )

    df = load_data()

    if df is None:
        return

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    while True:
        print(
            "\n" + "=" * 70,
            "ГЛАВНОЕ МЕНЮ",
            "=" * 70,
            f"\nТекущий набор данных: {len(df)} городов\n",
            "1. Показать информацию о данных",
            "2. Запустить K-means + метод локтя",
            "3. Запустить авторский агломеративный метод",
            "4. Запустить оба метода и сравнить",
            "0. Выход",
            sep="\n"
        )

        choice = input("\nВыберите действие: ").strip()

        if choice == "1":
            show_data_info(df)

        elif choice == "2":
            run_kmeans(df)

        elif choice == "3":
            run_custom(df)

        elif choice == "4":
            print("\nЗапуск K-means...")
            kmeans_result = run_kmeans(df)

            print("\nЗапуск авторского метода...")
            custom_result = run_custom(df)

            compare_methods(kmeans_result, custom_result)

        elif choice == "0":
            print("\nРабота программы завершена.")
            break

        else:
            print("\nНеизвестная команда.")


if __name__ == "__main__":
    main()
