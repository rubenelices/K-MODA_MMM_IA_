<div align="center">
  <img src="assets/kmoda_header.png" alt="K-MODA Marketing Mix Modeling" width="100%"/>
</div>

---

# K-MODA · Marketing Mix Modeling

**Trabajo universitario — Inteligencia Artificial · 3º Ingeniería Matemática · UAX**  
**Autor:** Rubén Elices · 2024-25

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://k-moda-ia-mmm.streamlit.app/)

---

## ¿Qué es este proyecto?

K-Moda es una empresa de moda española que invierte **13.8M€ al año** en publicidad distribuida en 8 canales (Paid Search, Social Paid, Video Online, Display, Email CRM, Radio Local, Exterior y Prensa). En un entorno post-cookie donde el tracking individual ya no es viable (GDPR 2018, iOS 14 en 2021, fin de cookies de terceros), la dirección necesita justificar ese presupuesto con datos agregados.

Este proyecto construye un pipeline completo de **Marketing Mix Modeling (MMM)** — una técnica econométrica que relaciona estadísticamente la inversión publicitaria con las ventas, sin depender de cookies ni píxeles de seguimiento.

---

## Estructura del pipeline

| Fase | Notebook | Contenido |
|------|----------|-----------|
| 1 | `01_etl.ipynb` | Carga, validación y rollup de 411.733 registros de ventas. Auditoría bruto vs. neto |
| 2 | `02_eda.ipynb` | Exploración de series temporales, estacionalidad, correlaciones y diagnóstico post-cookie |
| 3 | `03_feature_engineering.ipynb` | Transformaciones Lag y Adstock por canal. Construcción de `df_model` |
| 4 | `04_modeling.ipynb` | Entrenamiento ElasticNet con TimeSeriesSplit. Métricas MAPE y R² |
| 5 | `05_model_evaluation.ipynb` | Análisis de residuos, waterfall de atribución y tabla de mROI por canal |
| 6 | `06_simulator.ipynb` | Simulador de presupuesto con 3 escenarios estratégicos y optimización Simplex LP |
| — | `dashboard_streamlit.py` | Dashboard interactivo desplegado en Streamlit Cloud |

---

## Datos

- **Período:** 2020–2024 (261 semanas)
- **Ciudades:** Madrid, Barcelona, Valencia, Sevilla, Málaga, Zaragoza, Bilbao, Murcia, Palma, A Coruña
- **Canales publicitarios:** 8
- **Variable dependiente:** `venta_neta_sin_iva_eur` (nunca el importe bruto con IVA)
- **Registros fuente:** 411.733 líneas de venta + 20.960 registros de inversión

---

## Modelo

- **Algoritmo:** ElasticNet con `positive=True` (los coeficientes β no pueden ser negativos)
- **Validación:** `TimeSeriesSplit(n_splits=5)` — split temporal estricto, sin data leakage
- **Split:** Train 2020–2023 (208 semanas) / Test 2024 (53 semanas)
- **MAPE Train:** 5.82% · **MAPE Test:** 8.74%
- **R² Train:** 0.818 · **R² Test:** 0.574
- **Resultado:** Prensa purgada (β=0). 7 canales activos

### Transformaciones clave

**Adstock** — captura la memoria de marca (carry-over):
$$A_t = X_t + \alpha \cdot A_{t-1}$$

**Saturación logarítmica** — captura los rendimientos decrecientes:
$$S_t = \log\left(1 + \frac{A_t}{k}\right)$$

---

## Resultados de atribución

| Canal | mROI | Peso % | Inversión real |
|-------|------|--------|----------------|
| 🥇 Video Online | **14.8x** | 24.0% | 9.09M€ |
| 🥈 Display | **13.8x** | 11.9% | 4.80M€ |
| 🥉 Exterior | **13.7x** | 17.7% | 7.23M€ |
| Social Paid | 8.9x | 16.6% | 10.48M€ |
| Paid Search | 7.4x | 17.6% | 13.40M€ |
| Radio Local | 7.3x | 8.6% | 6.60M€ |
| Email CRM | 6.8x | 3.6% | 2.94M€ |
| Prensa | 0x ❌ | 0% | 5.45M€ |

---

## Optimización del presupuesto 2024

Usando programación lineal (Simplex LP) sobre los coeficientes β del modelo:

| Escenario | Inversión | Variación |
|-----------|-----------|-----------|
| Baseline 2023 | 13.8M€ | — |
| Conservador | 10M€ | −28% |
| **Óptimo** | **12M€** | **−13%** |

La propuesta óptima concentra el presupuesto en los canales con mROI > 13x, reduciendo el gasto total en 1.8M€ con mayor retorno que el baseline.

---

## Dashboard interactivo

🔗 **[k-moda-ia-mmm.streamlit.app](https://k-moda-ia-mmm.streamlit.app/)**

El dashboard incluye:
- Serie temporal real vs. predicho con métricas de validación
- Waterfall de descomposición de ventas por canal
- Simulador de presupuesto interactivo con sliders por canal
- Comparativa de 3 escenarios estratégicos
- Análisis de sensibilidad de coeficientes β

---

## Stack tecnológico

```
Python 3.13 · pandas · numpy · scikit-learn · plotly · streamlit
scipy · pyarrow · matplotlib · seaborn
```

---

## Estructura de archivos

```
K-MODA_MMM_IA_/
├── 01_etl.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_modeling.ipynb
├── 05_model_evaluation.ipynb
├── 06_simulator.ipynb
├── dashboard_streamlit.py
├── requirements.txt
├── data/
│   ├── df_model.parquet
│   ├── df_ventas_clean.parquet
│   └── df_inversion_clean.parquet
├── models/
│   ├── modelo_meta.pkl
│   ├── modelo_final.pkl
│   └── scaler.pkl
└── outputs/
    ├── tabla_atribucion.csv
    └── presupuesto_optimo_2024.csv
```

---

> *"Los datos no mienten: con el mismo presupuesto, bien distribuido, K-Moda puede crecer más. El MMM no es un modelo estadístico — es una ventaja competitiva."*
