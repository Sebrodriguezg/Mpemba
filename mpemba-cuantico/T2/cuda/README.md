# T2 en GPU (CUDA / T4) — evolución temporal densa con CuPy

Tercer escalón del escalado de la evolución temporal de T2, sobre **GPU T4** en
Google Colab: el *hotspot* (producto de matrices densas complejas $d\times d$ del
Liouvilliano matrix-free) se ejecuta con **CuPy**, que llama a **cuBLAS `zgemm`**.

> serie (1 hilo) → OpenMP (8 hilos, ~4×) → **CUDA T4 (cuBLAS)**

Reproduce *exactamente* el modelo de Ising disipativo de `../common/lindblad.hpp`,
así que el $D_{HS}$ de la GPU se cruza contra la CPU y contra el RK4 en C++
(`rk4_evolution`, $D_{HS}\approx0.3931849$ para $N=7$). Verificado en CPU: el
puerto NumPy reproduce ese valor a $2.6\times10^{-11}$.

## Flujo de trabajo

1. Sube `qmpe_t2_cuda_colab.ipynb` a [Google Colab](https://colab.research.google.com).
2. `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`.
3. `Entorno de ejecución → Ejecutar todo`.
4. Descarga lo que genera:
   - `Archivo → Descargar → .ipynb` (el **notebook ya ejecutado**, con salidas).
   - `resultados_t2_cuda.zip` (CSV + figuras, se descarga solo al final).
5. Coloca el notebook ejecutado en esta carpeta y el contenido del zip en
   `results/`. (Si me lo pasas, lo integro y, si quieres, añado una figura
   combinada **serie → OpenMP → GPU** uniendo estos datos con los de
   `../a_integracion_directa/results/serial_vs_parallel.csv`.)

## Qué hace el notebook

- **Validación cruzada triple** ($N=7$): $D_{HS}$ de GPU = CPU = C++ (la GPU no
  altera la física, solo acelera).
- **Benchmark A** — evolución RK4 completa, tiempo CPU vs GPU vs $N$ (malla
  $d=2^N$): la GPU pierde para $d$ pequeño (lanzamiento de kernels) y gana al
  crecer la malla. → `fig_t2_cuda_evolution.png`, `t2_cuda_evolution.csv`.
- **Benchmark B** — el *hotspot* aislado: GEMM complejo $d\times d$ (cuBLAS
  `zgemm`) CPU vs GPU, hasta $d=2048$. → `fig_t2_cuda_gemm.png`, `t2_cuda_gemm.csv`.

## Notas

- Herramienta: **CuPy** (preinstalado en Colab); `cupy.matmul` ⇒ cuBLAS. No hay
  que compilar nada.
- Precisión: `complex128` (doble), igual que los binarios C++, para que la
  validación sea estricta. En la T4 el FP64 va a 1/32 del FP32; aun así el GEMM
  denso gana claramente a $d$ grande.
- No requiere archivos del repo: el notebook es autocontenido.
