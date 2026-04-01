import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Este script carga la base de datos y conserva aquellos municipios con información suficiente.
# Posteriormente construye variables de exposición y prioridad.
# Finalmente agrupa municipios mediante clustering y genera un CSV con los municipios priorizados,
# su nivel de prioridad y su cluster. Además, guarda las figuras principales del análisis.

# Carpeta donde se encuentra este script
BASE_DIR = Path(__file__).resolve().parent

# Carga de archivo de base de datos
INPUT_XLSX = BASE_DIR / "Base de datos.xlsx"

# Archivo de salida
OUTPUT_CSV = BASE_DIR / "resultados_priorizacion_municipal.csv"

# Carpeta para guardar figuras
FIGURES_DIR = BASE_DIR / "figuras_priorizacion"

# Valor usado en el informe final por razones analíticas; si se desea usar el mejor k según silhouette,
# puede reemplazarse en la salida final del script.
REPORT_K = 4


def load_data(path: Path) -> pd.DataFrame:
    # Carga de archivo Excel y limpieza inicial
    df = pd.read_excel(path)

    count_cols = [
        "Monumentos Históricos",
        "Conjuntos Arquitectónicos",
        "Bienes Inmuebles con Valor Cultural",
        "Fichas sin Clasificación",
        "Total Monumentos",
        "Bien cultural Patrimonio mundial",
        "Bien mixto Patrimonio mundial",
        "Bien natural Patrimonio mundial",
        "Museo y Zona Arqueologica",
        "Museo",
        "Zona Arqueologica",
        "Sitios registrado",
    ]

    vulnerability_cols = [
        "Vulnerabilidad sociodemográfica",
        "Vulnerabilidad natural",
        "Vulnerabilidad normativa",
        "Vulnerabilidad iterada",
    ]

    # Los vacíos se sustituyen por 0
    for col in count_cols:
        df[col] = df[col].fillna(0)

    # Se conservan solo los registros con información completa en las variables de vulnerabilidad
    df = df.dropna(subset=vulnerability_cols).copy()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Construcción de variables sumarizadas (patrimonio mundial e infraestructura cultural)
    # y de índices compuestos
    df = df.copy()

    # Suma de patrimonio mundial
    df["patrimonio_mundial_total"] = (
        df["Bien cultural Patrimonio mundial"]
        + df["Bien mixto Patrimonio mundial"]
        + df["Bien natural Patrimonio mundial"]
    )

    # Suma de infraestructura cultural
    df["infraestructura_cultural_total"] = (
        df["Museo y Zona Arqueologica"] + df["Museo"] + df["Zona Arqueologica"]
    )

    # Variables para estimar exposición patrimonial
    exposure_features = [
        "Total Monumentos",
        "Sitios registrado",
        "infraestructura_cultural_total",
        "patrimonio_mundial_total",
    ]

    # Homologación con MinMaxScaler para variables
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[exposure_features])

    # Pesos para índice de exposición
    weights = np.array([0.45, 0.20, 0.20, 0.15])
    df["indice_exposicion"] = scaled.dot(weights)

    # Índice para priorización municipal
    # Aquí decidí dar más peso a la vulnerabilidad iterada para que la priorización
    # refleje más el componente de riesgo acumulado.
    df["indice_prioridad"] = 0.60 * df["Vulnerabilidad iterada"] + 0.40 * df["indice_exposicion"]

    # Clasificación para análisis
    df["nivel_prioridad"] = pd.qcut(
        df["indice_prioridad"],
        5,
        labels=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
    )

    return df


def cluster_municipios(df: pd.DataFrame, report_k: int = REPORT_K):
    # Esta función primero explora distintos valores de K para conocer cuál obtiene
    # el mejor silhouette score.
    # Aun así, la solución reportada en el proyecto usa k=4 por fines interpretativos.
    # Si se desea una salida totalmente guiada por silhouette, puede usarse best_k
    # en lugar de REPORT_K dentro de la salida final del script.
    df = df.copy()

    features = [
        "Vulnerabilidad sociodemográfica",
        "Vulnerabilidad natural",
        "Vulnerabilidad normativa",
        "indice_exposicion",
    ]

    # Estandarización: media 0 y desviación estándar 1
    X = StandardScaler().fit_transform(df[features])

    best_score = -1
    best_k = None

    # Prueba exploratoria con diferentes números de clusters
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    # Modelo final que se usará para las salidas del script y las figuras
    report_model = KMeans(n_clusters=report_k, random_state=42, n_init=20)
    df["cluster"] = report_model.fit_predict(X)

    # PCA para visualización e interpretación
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    report_score = silhouette_score(X, df["cluster"])

    print(f"Mejor número de clusters según silhouette: {best_k} (score={best_score:.3f})")
    print(f"Solución utilizada para el informe y las figuras: k={report_k} (score={report_score:.3f})")

    return df, best_k, best_score, report_k, report_score


def plot_top_entidades(df: pd.DataFrame, output_dir: Path) -> None:
    # Figura 1. Entidades con mayor prioridad patrimonial promedio
    state_priority = (
        df.groupby("EDO", dropna=False)["indice_prioridad"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(state_priority.index.astype(str), state_priority.values)
    plt.xlabel("Índice de prioridad promedio")
    plt.title("Top 10 entidades por prioridad patrimonial promedio")
    plt.tight_layout()
    plt.savefig(output_dir / "figura_1_top_entidades_prioridad.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_municipios(df: pd.DataFrame, output_dir: Path) -> None:
    # Figura 2. Municipios con mayor índice de prioridad patrimonial
    municipios = df.copy()
    municipios["municipio_etiqueta"] = (
        municipios["NOM"].astype(str) + " (" + municipios["EDO"].astype(str) + ")"
    )

    top_municipios = (
        municipios[["municipio_etiqueta", "indice_prioridad"]]
        .sort_values("indice_prioridad", ascending=False)
        .head(10)
        .sort_values("indice_prioridad", ascending=True)
    )

    plt.figure(figsize=(9, 5.5))
    plt.barh(top_municipios["municipio_etiqueta"], top_municipios["indice_prioridad"])
    plt.xlabel("Índice de prioridad")
    plt.title("10 municipios con mayor prioridad patrimonial")
    plt.tight_layout()
    plt.savefig(output_dir / "figura_2_top_municipios_prioridad.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_exposicion_vs_vulnerabilidad(df: pd.DataFrame, output_dir: Path) -> None:
    # Figura 3. Relación entre exposición y vulnerabilidad iterada
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(
        df["indice_exposicion"],
        df["Vulnerabilidad iterada"],
        alpha=0.5,
        s=12,
    )
    plt.xlabel("Índice de exposición patrimonial")
    plt.ylabel("Vulnerabilidad iterada")
    plt.title("Relación entre exposición y vulnerabilidad")
    plt.tight_layout()
    plt.savefig(output_dir / "figura_3_exposicion_vs_vulnerabilidad.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_clusters(df: pd.DataFrame, output_dir: Path) -> None:
    # Figura 4. Visualización PCA de la segmentación municipal
    plt.figure(figsize=(7, 6))
    clusters = sorted(df["cluster"].dropna().unique())

    for cluster_id in clusters:
        subset = df[df["cluster"] == cluster_id]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            s=15,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Segmentación municipal (PCA de variables normalizadas)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figura_4_pca_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figures(df: pd.DataFrame, output_dir: Path) -> None:
    # Genera las figuras principales utilizadas en el informe
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_top_entidades(df, output_dir)
    plot_top_municipios(df, output_dir)
    plot_exposicion_vs_vulnerabilidad(df, output_dir)
    plot_pca_clusters(df, output_dir)

    print(f"Figuras guardadas en: {output_dir}")


def main():
    df = load_data(INPUT_XLSX)
    df = engineer_features(df)
    df, best_k, best_score, report_k, report_score = cluster_municipios(df)

    # Se exporta el resultado completo para su consulta posterior
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Se generan las figuras del análisis
    generate_figures(df, FIGURES_DIR)

    cols = [
        "EDO",
        "NOM",
        "indice_prioridad",
        "nivel_prioridad",
        "cluster",
        "Vulnerabilidad iterada",
        "indice_exposicion",
        "Total Monumentos",
        "Sitios registrado",
        "patrimonio_mundial_total",
    ]

    # Visualiza 15 municipios con mayor prioridad
    print(df[cols].sort_values("indice_prioridad", ascending=False).head(15).to_string(index=False))
    print(f"\nArchivo guardado en: {OUTPUT_CSV}")
    print(f"Mejor k exploratorio: {best_k}")
    print(f"Silhouette score exploratorio: {best_score:.3f}")
    print(f"k usado en salidas y figuras: {report_k}")
    print(f"Silhouette score de la solución reportada: {report_score:.3f}")


if __name__ == "__main__":
    main()