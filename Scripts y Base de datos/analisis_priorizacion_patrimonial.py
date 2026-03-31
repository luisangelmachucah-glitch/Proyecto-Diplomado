import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Este script carga la base de datos y conserva aquellos municipios con información suficiente.
# Posteriormente construye variables de exposición y prioridad
# Finalmente agrupa municipios mediante clustering y da resultado, siendo un csv con los municipios priorizados, su nivel prioridad y su cluster.

# Carpeta donde se encuentra este script
BASE_DIR = Path(__file__).resolve().parent

# Carga de archivo de base de datos
INPUT_XLSX = BASE_DIR / "Base de datos.xlsx"

# Archivo de salida
OUTPUT_CSV = BASE_DIR / "resultados_priorizacion_municipal.csv"


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
    # Construccion de variables sumarizadas (patrimonio mundial, e infraestructura cultural) y de índices compuestos
    # El índice de exposición se calcula a partir de cuatro variables normalizadas con MinMaxScaler para homologarlas en [0, 1] para ponderar su importancia analítica
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

    # Homologacion con MinMaxScaler para variables
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[exposure_features])

    # Pesos para índice de exposición
    weights = np.array([0.45, 0.20, 0.20, 0.15])
    df["indice_exposicion"] = scaled.dot(weights)

    # Índice para priorización municipal
    # Se da más peso a la vulnerabilidad iterada por fines de calculo propio
    df["indice_prioridad"] = 0.60 * df["Vulnerabilidad iterada"] + 0.40 * df["indice_exposicion"]

    # Clasificación para analisis
    df["nivel_prioridad"] = pd.qcut(
        df["indice_prioridad"],
        5,
        labels=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
    )

    return df


def cluster_municipios(df: pd.DataFrame) -> pd.DataFrame:
    # El comando agrupa municipios con KMeans a partir de variables de vulnerabilidad y exposición, estandarizando las variables, teniendo distintos de valores K para escoger el mejor con silhoute score
    # Finalmente, se asigna el cluster final a cada municipio, y se reducen dimensiones con PCA para visualización
    df = df.copy()

    features = [
        "Vulnerabilidad sociodemográfica",
        "Vulnerabilidad natural",
        "Vulnerabilidad normativa",
        "indice_exposicion",
    ]

    # Estandarización: media 0 y desviación estándar 1
    X = StandardScaler().fit_transform(df[features])

    best_model = None
    best_score = -1
    best_k = None

    # Prueba con diferentes números de clusters
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        # Modelo con mejor valor de silhouette
        if score > best_score:
            best_model = model
            best_score = score
            best_k = k

    # Etiqueta de cluster
    df["cluster"] = best_model.predict(X)

    # PCA para visualización e interpretación espacial
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    print(f"Mejor número de clusters según silhouette: {best_k} (score={best_score:.3f})")
    return df


def main():
    df = load_data(INPUT_XLSX)
    df = engineer_features(df)
    df = cluster_municipios(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

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


if __name__ == "__main__":
    main()
