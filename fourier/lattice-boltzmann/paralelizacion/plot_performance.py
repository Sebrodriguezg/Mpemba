import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    csv_file = "benchmark_results.csv"
    
    try:
        # Leer los resultados generados por el script de bash
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: No se encontró '{csv_file}'. Ejecuta ./run_benchmark.sh primero.")
        sys.exit(1)

    # Extraer el tiempo del baseline serial asegurando que sea float
    try:
        t_serial = float(df[df['Method'] == 'Serial']['Time'].values[0])
    except IndexError:
        print("Error: No se encontraron datos de la ejecución Serial.")
        sys.exit(1)

    # Filtrar datos por tecnología
    df_omp = df[df['Method'] == 'OpenMP']
    df_mpi = df[df['Method'] == 'MPI']
    
    # Extraer CUDA (si existe) asegurando que el tiempo sea float
    df_cuda = df[df['Method'] == 'CUDA']
    t_cuda = float(df_cuda['Time'].values[0]) if not df_cuda.empty else None

    # Extraer los tiempos y forzarlos a números
    time_omp = pd.to_numeric(df_omp['Time']).values
    time_mpi = pd.to_numeric(df_mpi['Time']).values

    # Calcular Speedups (S = T_serial / T_paralelo)
    speedup_omp = t_serial / time_omp
    speedup_mpi = t_serial / time_mpi

    # SOLUCIÓN AL ERROR: Convertir explícitamente los hilos a números
    threads_omp = pd.to_numeric(df_omp['Threads']).values
    threads_mpi = pd.to_numeric(df_mpi['Threads']).values

    # Calcular Eficiencia Computacional (E = S / p)
    eff_omp = speedup_omp / threads_omp
    eff_mpi = speedup_mpi / threads_mpi

    # --- Configuración de las Gráficas ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Speedup
    ax1.plot(threads_omp, threads_omp, 'k--', label='Ideal (Lineal)')
    
    if not df_omp.empty:
        ax1.plot(threads_omp, speedup_omp, 'bo-', linewidth=2, markersize=8, label='OpenMP')
    if not df_mpi.empty:
        ax1.plot(threads_mpi, speedup_mpi, 'ro-', linewidth=2, markersize=8, label='MPI')
        
    if t_cuda:
        speedup_cuda = t_serial / t_cuda
        ax1.axhline(speedup_cuda, color='green', linestyle='-', linewidth=2, 
                    label=f'CUDA GPU (Speedup: {speedup_cuda:.1f}x)')

    ax1.set_title('Speedup vs Recursos Computacionales', fontsize=14)
    ax1.set_xlabel('N° Hilos / Procesos MPI', fontsize=12)
    ax1.set_ylabel('Speedup ($T_{serial} / T_{paralelo}$)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(threads_omp)

    # Panel 2: Eficiencia
    ax2.plot(threads_omp, [1.0]*len(threads_omp), 'k--', label='Ideal (100%)')
    
    if not df_omp.empty:
        ax2.plot(threads_omp, eff_omp, 'bo-', linewidth=2, markersize=8, label='OpenMP')
    if not df_mpi.empty:
        ax2.plot(threads_mpi, eff_mpi, 'ro-', linewidth=2, markersize=8, label='MPI')

    ax2.set_title('Eficiencia Computacional', fontsize=14)
    ax2.set_xlabel('N° Hilos / Procesos MPI', fontsize=12)
    ax2.set_ylabel('Eficiencia (Speedup / $p$)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(threads_omp)
    ax2.set_ylim(0, 1.1) # La eficiencia ideal es 1.0 (100%)

    plt.tight_layout()
    nombre_salida = 'rendimiento_hpc.png'
    plt.savefig(nombre_salida, dpi=300)
    print(f"Gráfica de rendimiento guardada exitosamente como '{nombre_salida}'.")

if __name__ == "__main__":
    main()
