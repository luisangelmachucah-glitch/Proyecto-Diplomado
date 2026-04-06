# Priorización municipal del riesgo para el patrimonio cultural en México
# https://youtu.be/n_BvGiRLNgg

Este repositorio compila los insumos principales del proyecto final del Diplomado en Ciencia de Datos.  
El objetivo es construir una metodología de priorización municipal que combine:

- conteos patrimoniales obtenidos mediante web scraping del catálogo de monumentos históricos del INAH;
- variables de vulnerabilidad sociodemográfica, natural y normativa;
- un índice compuesto de prioridad;
- y una segmentación municipal con técnicas de clustering.

## Objetivo del proyecto

Generar una base analítica que ayude a identificar municipios donde la combinación entre **exposición patrimonial** y **vulnerabilidad** sugiera una mayor prioridad de atención, conservación o diagnóstico

## Estructura del repositorio

- Base de datos.xlsx
  Base municipal integrada. Incluye los conteos patrimoniales y las variables de vulnerabilidad empleadas en el análisis

- scrape_INAH_v010.py
  Script de scraping que automatiza la consulta del catálogo de monumentos históricos del INAH por estado y municipio

- analisis_priorizacion_patrimonial.py
  Script principal de análisis. Limpia la base, construye variables agregadas, calcula el índice de exposición, genera el índice de prioridad y aplica clustering con KMeans

- Informe_tecnico_proyecto_final.pdf
  Versión en PDF del informe técnico del proyecto

## Fuente de datos

La solución parte de dos insumos principales:

1. Datos patrimoniales extraídos mediante scraping del catálogo del INAH
2. Variables de vulnerabilidad integradas previamente a nivel municipal

## Principales librerías utilizadas

- pandas
- numpy
- scikit-learn
- selenium
- openpyxl
- pathlib

## Salidas principales

- ranking estatal por prioridad patrimonial promedio
- ranking municipal por índice de prioridad
- relación entre exposición patrimonial y vulnerabilidad iterada
- perfiles municipales obtenidos con clustering

## Consideraciones

- Los conteos patrimoniales dependen de la información disponible en el catálogo público consultado
- La priorización se construye con ponderaciones explícitas, por lo que puede ajustarse en futuros ejercicios

## Reproducibilidad

Para replicar el proyecto se recomienda conservar juntos:

- la base en Excel
- el script de scraping
- el script de análisis
- y el informe técnico

## Autoría

Proyecto elaborado por Luis Ángel Machuca Herrera como parte del trabajo final del Diplomado en Ciencia de Datos.
