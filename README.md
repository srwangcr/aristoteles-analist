# Proyecto 2 — Detección de errantes en ruido de imágenes astronómicas

[English version: README.en.md](README.en.md)

Motor modular y reproducible para detectar objetos con movimiento lineal o cuasi-lineal ocultos bajo el ruido de series temporales FITS. El pipeline cubre simulación calibrada con verdad terrestre, preprocesamiento astrométrico, *shift-and-stack* multi-hipótesis, validación de trazas y evaluación científica.

## Instalación

Requiere Python 3.12+ (probado con 3.14).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/pytest -q
```

Dependencias nucleares: NumPy, SciPy, Astropy (FITS/WCS) y Polars (catálogos y métricas). El kernel de apilamiento está en NumPy puro; si el perfilado lo justifica, la etapa `ShiftStackEngine.stack` es el único punto candente a portar a Rust/PyO3.

## Estructura

```
src/errantes/
  types.py        # configuraciones, Wanderer, Candidate, GroundTruth, SimulatedCube
  physics.py      # PSF Gaussiana/Moffat, Poisson+Gaussiano, flujo↔SNR↔magnitud
  simulator.py    # cubo sintético, inyección de verdad terrestre, E/S FITS
  preprocess.py   # malla MAD, correlación de fase, imagen maestra y residuos
  detection.py    # ShiftStackEngine, umbral kσ, componentes, centroides
  validation.py   # RANSAC lineal, consistencia fotométrica, rechazo de cósmicos
  pipeline.py     # orquestación extremo a extremo
  metrics.py      # matching húngaro, Precision/Recall/F1, ROC/PR, benchmark
benchmarks/
  run_benchmark.py  # CLI de sensibilidad y escalado
  results/          # Parquet/JSON con resultados empíricos
tests/              # 24 tests unitarios y de integración
```

## Modelo físico

Para el frame $i$, el número esperado de electrones por píxel es

$$
\mu_i(x,y) = \underbrace{b_0 + g_x(x-c_x) + g_y(y-c_y) + \delta_i}_{\text{cielo con gradiente y deriva}}
+ \sum_s F_s\,P_i(x-x_s-\Delta x_i,\,y-y_s-\Delta y_i)
+ F_{\rm obj}\,P_i(x-x_i,\,y-y_i) + C_i(x,y),
$$

donde $\delta_i\sim\mathcal{N}(0,\sigma_{\rm drift}^2)$, $\Delta x_i,\Delta y_i\sim\mathcal{N}(0,\sigma_{\rm jitter}^2)$ representan el *jitter* de guiado de estrellas, $C_i$ son rayos cósmicos (píxeles de un solo frame) y columnas defectuosas (sesgo aditivo constante por columna). La imagen digitalizada es

$$
D_i(x,y) = \frac{\mathrm{Poisson}\!\left(\mu_i(x,y)\right) + \mathcal{N}(0,\sigma_{\rm read}^2)}{g},
$$

con $g$ el *gain* en e⁻/ADU. La dispersión local en electrones es $\sigma_{\rm loc}=\sqrt{\mu_{\rm cielo}+\sigma_{\rm read}^2}$.

PSF Gaussiana: $P=\exp[-r^2/(2\sigma^2)]$, normalizada a suma 1, con pico $P(0)=1/(2\pi\sigma^2)$. PSF de Moffat: $P=(1+r^2/\alpha^2)^{-\beta}$, con $\alpha=2\sigma\sqrt{2^{1/\beta}-1}$ para igualar la FWHM Gaussiana. El flujo inyectado para un SNR deseado por frame se deriva de

$$
\mathrm{SNR}_{\rm frame}=\frac{F_{\rm obj}P(0)}{\sqrt{\mu_{\rm cielo}+\sigma_{\rm read}^2}}
\quad\Longrightarrow\quad
F_{\rm obj}=\frac{\mathrm{SNR}_{\rm frame}\,\sigma_{\rm loc}}{P(0)}.
$$

Todas las magnitudes son instrumentales: $m = m_{\rm zp}-2.5\log_{10}(F/g)$.

## Algoritmo

1. **Fondo y estrellas.** Malla 2D de medianas con *sigma-clipping* iterativo (mediana + $1.4826\,\mathrm{MAD}$) e interpolación bilineal. La alineación usa correlación de fase con zero-padding y ajuste parabolico del pico, precision sub-píxel. La imagen maestra es la mediana temporal de los frames alineados y normalizados por el flujo de estrellas brillantes. Se resta **solo en el soporte de fuentes estacionarias** (máscara dilatada de la imagen maestra): restar la mediana en todo el campo auto-sustraería errantes lentos que permanecen varios frames sobre el mismo píxel. Cuando se conocen los *seeing* por frame se suaviza la referencia con $\sigma_{\rm extra}=\sqrt{\sigma_i^2-\sigma_{\rm ref}^2}$ antes de restar.

2. **Shift-and-stack (Track-Before-Detect).** Para cada hipótesis $\vec v=(v_x,v_y)$ se suman los residuos desplazados al frame de referencia $t_r$:

$$
S_{\vec v}(x,y)=\sum_{i=0}^{N-1} R_i\!\left(x+v_x(t_i-t_r),\,y+v_y(t_i-t_r)\right).
$$

En el dominio de Fourier, con $k_x,k_y$ en índices de frecuencia y $s_x=-v_x(t_i-t_r)$, $s_y=-v_y(t_i-t_r)$:

$$
\hat R_i^{\,s}(k_x,k_y)=\hat R_i(k_x,k_y)\,
\exp\!\left[-2\pi i\left(\frac{k_xs_x}{W}+\frac{k_ys_y}{H}\right)\right].
$$

Los $N$ FFT directos se calculan una sola vez; cada hipótesis cuesta $N$ inversas. Se aplica un filtro matched Gaussiano ($\sigma=0.8$ px) al mapa apilado antes de umbralizar a $k\sigma$ (MAD). Se etiquetan componentes conexas 3×3 y se calcula el centroide ponderado sub-píxel.

3. **Validación.** Se mide el flujo de apertura y el centroide en cada frame sobre la traza pronosticada; se ajusta $(x(t),y(t))$ con RANSAC lineal y se descartan candidatos con RMS > 0.9 px, fracción de energía en un solo frame > 0.55, CV fotométrico > 0.65 o menos del 70 % de frames dentro del campo. Los rayos cósmicos concentran toda su energía en un único frame y son eliminados por este filtro.

4. **Métricas.** Asociación óptima candidato↔verdad por el algoritmo húngaro (tolerancias 2.5 px en posición y 0.45 px/frame en velocidad), Precision, Recall, F1 y curvas ROC/PR al barrer el SNR del candidato.

### Complejidad

| Etapa | Coste |
|---|---|
| Fondo + sustracción | $O(NHW)$ |
| Alineamiento | $O(NHW\log(HW))$ |
| FFT de residuos (una vez) | $O(NHW\log(HW))$ |
| Barrido de $M$ velocidades | $O(M\,N\,HW\log(HW))$ |
| Memoria | $O(NHW + HW)$ además de los $N$ espectros complejos |

Con $M=\left(2v_{\max}/\Delta v+1\right)^2$: $v_{\max}=1.5$, $\Delta v=0.25$ dan $M=169$; $\Delta v=0.125$ dan $M=625$.

## Uso rápido

```python
from errantes import simulate_default_cube, run_pipeline
from errantes.metrics import match_detections, roc_pr_curve

cube = simulate_default_cube(snr_frame=2.0, seed=1234)
result = run_pipeline(cube)

for candidate in result.candidates:
    print(candidate.x, candidate.y, candidate.vx, candidate.vy, candidate.snr)

matched = match_detections(cube.truth.positions, result.candidates, ref_frame=5)
print(matched.as_dict())
```

La verdad terrestre se exporta con `cube.truth.write("truth/")` a `objects.json`, `objects.parquet`, `positions.json` y `positions.parquet`. `write_fits_cube(cube, "cube.fits")` escribe el cubo con WCS lineal y una extensión `TRUTH` con $(t,x,y,v_x,v_y,\mathrm{SNR},F)$ por objeto y frame.

## Resultados empíricos

Benchmark con 10 semillas por SNR, cubos de $12\times96\times96$ px, $v=(0.37,-0.21)$ px/frame, estrellas, *seeing* variable, *jitter*, columnas defectuosas y ~3 rayos cósmicos/frame. CPU AMD Ryzen 5 3400G, NumPy 2.5.5 / SciPy 1.18.1, una sola hebra. Datos completos en `benchmarks/results/`.

| SNR/frame | Precision | Recall | F1 | FP medio | Tiempo/cubo (s) |
|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.60 | 0.60 | 0.60 | 0.0 | 1.25 |
| 2.0 | 0.85 | 0.90 | 0.87 | 0.1 | 1.35 |
| 3.0 | 0.90 | 0.90 | 0.90 | 0.0 | 1.29 |
| 5.0 | 1.00 | 1.00 | 1.00 | 0.0 | 1.31 |

Escalado del barrido (mismo cubo, media de 3 semillas):

| $\Delta v$ (px/frame) | $M$ | Tiempo (s) |
|---:|---:|---:|
| 0.5 | 49 | 0.70 |
| 0.25 | 169 | 1.32 |
| 0.125 | 625 | 3.20 |

El tiempo crece aproximadamente lineal con $M$, como predice $O(M N H W\log(HW))$. El apilamiento integra $\sqrt{N}\approx3.46$ veces la señal; el filtro matched y la sustracción parcial de estrellas reducen la ganancia efectiva observada a un SNR apilado de ~5–9 para $\mathrm{SNR}_{\rm frame}=2$.

Reproducir:

```bash
.venv/bin/python benchmarks/run_benchmark.py --scaling
.venv/bin/pytest -q
.venv/bin/mypy src/errantes
```

Verificación incluida: recuperación de errantes con $\mathrm{SNR}_{\rm frame}=2.0$, error de posición < 1.5 px y de velocidad < 0.5 px/frame; **100 % de rechazo de rayos cósmicos** en 4 cubos con tasa 12 cósmicos/frame (≈48 inyecciones) y un test unitario dedicado al pico de un solo frame.

## Limitaciones y trabajo futuro

- La sustracción de estrellas enmascara el soporte de las fuentes fijas: un errante que cruza exactamente el núcleo de una estrella brillante puede perder señal en esos frames.
- El desplazamiento en Fourier asume bordes periódicos; los candidatos a menos de 2 px del borde se descartan en la validación.
- La rejilla de velocidades es regular; una búsqueda jerárquica (gruesa → fina) reduciría el coste para $v_{\max}$ grande.
- Perfilado actual: ~98 % del tiempo en las $M\times N$ inversas FFT. El siguiente paso natural es un kernel Rust/PyO3 con FFT planificados y acumulación en sitio, manteniendo la API de `ShiftStackEngine`.
