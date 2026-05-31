# Datos estandarizados del efecto Mpemba

Aislamiento y estandarización de los datos más relevantes de los **3 papers** del
repositorio `mpemba_data/`, para observar el efecto Mpemba en tres dominios físicos.

## Origen de los datos

| Fuente | Dominio | Archivo original | Observable aislada |
|---|---|---|---|
| **Kumar et al. 2022, PNAS** | Clásico / experimental (coloide en trampa óptica) | `Excel FIles_Graphical Data.xlsx`, hoja `Fig5` | Distancia al equilibrio `D(t)` para 7 temperaturas iniciales `T0` |
| **Hallstadius & Burridge 2020, Proc. R. Soc. A** | Clásico / experimental (congelación de agua) | `rspa20190829_si_001.xlsx`, hoja `Test` | Temperatura `T(°C)` vs tiempo; superficie lisa vs rugosa; 21 ensayos |
| **Turkeshi et al. 2024, arXiv** | Cuántico / numérico (circuito aleatorio) | `TN_TF_N512NA16.csv` | Asimetría de entrelazamiento `ΔS̃₂ = EAQ − EA` para 4 ángulos `θ` |

## Esquema común (`*.csv`)

Formato *tidy* de 9 columnas:

```
source, system, domain, protocol, protocol_value, x_name, x, obs_name, obs
```

- `protocol` / `protocol_value`: la condición inicial que se varía (T0, ángulo θ, ensayo+superficie).
- `x` / `x_name`: la variable independiente (tiempo, en las unidades nativas de cada paper).
- `obs` / `obs_name`: la "distancia al estado estacionario" cuya relajación revela el efecto.

Archivos generados:
- `kumar2022_colloid.csv`, `hallstadius2020_water.csv`, `turkeshi2024_quantum.csv`
- `mpemba_all_standardized.csv` (los tres concatenados)

## Cómo reproducir

```bash
python standardize.py     # aísla y estandariza -> CSVs
python plot_mpemba.py      # genera las figuras PNG
```

Dependencias: `pandas`, `openpyxl`, `numpy`, `matplotlib`.

## Figuras

- **`mpemba_comparison.png`** — 3 paneles (un paper por panel). El efecto Mpemba se
  ve como el **cruce de curvas**: una condición inicial más alejada del equilibrio
  lo alcanza antes que otra más cercana. En el panel cuántico, `θ=1.2` parte como el
  estado más asimétrico pero restaura la simetría más rápido y cruza por debajo del resto.
- **`mpemba_water_diagnostic.png`** — diagnóstico cuantitativo del agua: tiempo en
  alcanzar −5 °C para la muestra **caliente** vs la **fría** de cada ensayo
  `DifferentTemp`. Cuando la caliente congela primero (barra roja < azul) ⇒ efecto Mpemba.

## Hallazgo

El efecto Mpemba se observa de forma nítida y robusta en el coloide (Kumar) y en el
sistema cuántico (Turkeshi). En el agua (Hallstadius) aparece solo en una parte de los
ensayos `DifferentTemp` y no en los `DiffTemp` con diferencias de temperatura extremas:
es un efecto **frágil**, dependiente de los sitios de nucleación (superficie rugosa),
en línea con la conclusión del propio artículo.
