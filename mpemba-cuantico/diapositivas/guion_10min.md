# Guion — Efecto Mpemba cuántico en HPC (10 min)

Charla conjunta. **Juan** (teal) habla sus secciones; **Sebastián** (naranja) las suyas.
Tiempo total ≈ 10 min (dentro de la expo de 30 min del repo). Cada bloque indica
el orador, la diapositiva y los puntos a decir (lenguaje natural, no leer literal).

> Reparto: **Juan** → intro+matemática, T1, T3b, conclusiones. **Sebastián** → T2a, T2b, T3, CUDA, relevancia macro.

---

## 0 · Apertura — *Título* · **Juan** · 0:15
- "Vamos a contar cómo llevamos el **efecto Mpemba cuántico** a HPC: cuatro tareas
  numéricas, cada una con su versión serial y paralela, validadas y con escalado."
- Señalar el reparto de colores (teal Juan / naranja Sebastián).

## 1 · De lo clásico a lo cuántico — *La anomalía* · **Juan** · 0:45
- "El Mpemba clásico: a veces el agua **más caliente se congela antes**. La idea
  clave no es la temperatura, sino que **la curva que parte más lejos del equilibrio
  lo alcanza primero** — un *cruce* de curvas de relajación."
- "En sistemas cuánticos abiertos reaparece igual: dos preparaciones, y la que parte
  lejos **adelanta** a la cercana en un tiempo t*." (señalar la figura del cruce)

## 1 · *Matemática: GKSL y autovalores* · **Juan** · 0:55
- "La dinámica es la **ecuación GKSL** (Lindblad): parte coherente más disipador."
- "Diagonalizando el **Liouvilliano** obtenemos sus autovalores λ_k: la relajación es
  una **suma de modos** que decaen, dominada por el **modo lento** λ₂."
- "El **criterio de Mpemba** es puro álgebra: si el solapamiento con el modo lento
  **a₂ = Tr(l₂† ρ₀) = 0**, la preparación *salta* el modo lento y relaja a la tasa
  rápida → de ahí el cruce. Lo permite la **no-normalidad** de L."
- *(Transición)* "Y para eso, lo primero es calcular esos autovalores. Tarea 1."

## 2 · T1 Modos lentos — *Método* · **Juan** · 0:40
- "Queremos λ₂, λ₃… **sin diagonalizar** una matriz 4^N × 4^N."
- "Truco: **Arnoldi sobre el propagador** e^{τL}. Como los λ negativos se vuelven los
  autovalores *dominantes* del propagador, Arnoldi los saca primero — *faster than
  the clock*. La acción de L es **matrix-free** (RK4)."

## 2 · T1 — *Núcleo serial∥paralelo* · **Juan** · 0:30
- "El hotspot es el producto de matrices. **Serial**: todas las filas. **Paralelo**:
  cada rango MPI calcula **su bloque de filas** (OpenMP dentro), y se reúne con
  **MPI_Allgatherv** — paralelismo de datos."

## 2 · T1 — *Resultados* · **Juan** · 0:30
- "Validado: error en autovalores hasta 10⁻¹³. Escalado ~2.9× (OpenMP/MPI): el SpMV
  está **acoplado** y es *memory-bound*, por eso satura."
- "Importante: T1 nos da el λ₂ y el a₂ que **explican el cruce**." *(pasa a Sebastián)*

## 3 · T2a Evolución RK4 — *Método* · **Sebastián** · 0:40
- "Ahora evolucionar ρ(t) en el tiempo. Vía directa: **RK4** sobre dρ/dt = L[ρ],
  aplicando L **matrix-free** (sin formar la matriz gigante)."
- "Los pasos dependen del anterior → no se paraleliza *entre* pasos; el paralelismo
  está **dentro** del paso, en el producto de matrices → **OpenMP** (memoria
  compartida, sin comunicación)."

## 3 · T2a — *Núcleo serial∥paralelo* · **Sebastián** · 0:30
- "El núcleo es esto: una **sola directiva**, `#pragma omp parallel for`, reparte las
  filas entre hilos. El mismo binario, cambiando OMP_NUM_THREADS, da serie vs paralelo."

## 3 · T2a — *Resultados* · **Sebastián** · 0:35
- "Speedup que **crece con la malla** hasta ~4× (sublineal, *bandwidth-bound*)."
- "Y aquí está el efecto: **la preparación que parte lejos (roja) adelanta a la
  cercana (azul) en t\*≈0.69**." (señalar el cruce)

## 4 · T2b Krylov — *Método + resultado central* · **Sebastián** · 0:45
- "Misma evolución, otra vía: **acción del exponencial** e^{τL} por Krylov (Arnoldi +
  exponencial de una matriz pequeña). Mismo hotspot → misma paralelización."
- "La ventaja: **a igual trabajo, ~25× más preciso** que RK4, con pasos de tiempo
  grandes e incondicionalmente estable." (señalar la curva de comparación)

## 4 · T2b — *Cruce* · **Sebastián** · 0:20
- "Y reproduce el **mismo cruce, t\*≈0.69**, idéntico a RK4 → el efecto es físico, no
  un artefacto del integrador."

## 5 · T3 Trayectorias — *Método + núcleo* · **Sebastián** · 0:40
- "Para **muchos cuerpos**, ρ (de tamaño 4^N) no cabe. Solución: **trayectorias** —
  propagar M estados puros (tamaño 2^N) y promediar. Saltos cuánticos."
- "Cada trayectoria es **independiente** → MPI reparte las M, y **una sola
  MPI_Reduce** promedia. Sin comunicación intermedia."

## 5 · T3 — *Resultados* · **Sebastián** · 0:30
- "Escalado **casi ideal, ~5.6× y constante** — contraste total con el OpenMP
  sublineal de T2. **El patrón de cómputo dicta el paradigma.**"
- "Y otra vez el **cruce de Mpemba, t\*≈0.75**, ahora con el método estocástico."

## 5 · CUDA T4 — *GPU* · **Sebastián** · 0:30
- "Tercer escalón: el producto denso va perfecto a **GPU (cuBLAS) en una T4**.
  Pierde para tamaños pequeños, pero **gana ~5×** a N≥7 (N=9: 943 s → 190 s)."
- "Validación triple GPU = CPU = C++. El notebook ejecutado y los resultados están en
  el repo." *(pasa a Juan)*

## 6 · T3b Redes tensoriales — *Método* · **Juan** · 0:40
- "La otra ruta a muchos cuerpos: **redes tensoriales**. ρ como un **superket MPS** y
  la ecuación maestra por **TEBD** — compuertas locales + truncación SVD a χ. Coste
  **polinómico en χ**, no exponencial en N."
- "Paralelizo por **hilos**: en una capa, los bonds pares (o impares) son disjuntos;
  sus SVD son independientes y numpy **libera el GIL**."

## 6 · T3b — *Núcleo serial∥paralelo* · **Juan** · 0:20
- "Serial: bucle sobre bonds. Paralelo: `executor.map` lanza las compuertas disjuntas
  en hilos. Paralelismo **estructurado** — intermedio entre T1 y las trayectorias."

## 6 · T3b — *Resultados* · **Juan** · 0:30
- "Validado vs exacto (error baja con χ), escala ~3.6× en hilos, y llega a **N=32** —
  un espacio de 10¹⁹, imposible en denso — en segundos."

## 7 · Conclusiones — **Juan** · 0:45
- "**HPC:** el patrón de cómputo manda. Acoplado (T1, ~3×) < estructurado (T3b, ~3.6×)
  < independiente (trayectorias, ~5.6× casi ideal); y la evolución densa vuela en GPU."
- "**Física:** el Mpemba cuántico es **geometría espectral** (no-normalidad + a₂=0);
  lo hemos **reproducido con los tres métodos de evolución** (t\*≈0.69–0.75),
  validado de forma cruzada, y escalado a muchos cuerpos." *(pasa a Sebastián)*

## 8 · Relevancia macro/meso — **Sebastián** · 0:35
- "Y cierra el círculo: **la firma es universal** — el mismo mecanismo espectral
  explica el cruce del macro clásico al cuántico."
- "Lo que aporta el enfoque cuántico: un **criterio operacional** (a₂=0) y las
  herramientas HPC para **predecir y diseñar** el efecto — con aplicaciones en
  **termometría cuántica** y enfriamiento acelerado."
- "Gracias." *(cierre)*

---

### Notas de tiempo
- Suma ≈ 10:00. Si vas justo: recorta la lectura de ecuaciones (slide matemática) y
  la slide CUDA (es un *bonus*).
- Si sobra tiempo: detente más en los **dos cruces** (T2a y T3) y en el contraste de
  escalado OpenMP vs MPI (es el mensaje HPC central).
- Punto de aplauso/énfasis: "**el patrón de cómputo dicta el paradigma**" (conclusión).
