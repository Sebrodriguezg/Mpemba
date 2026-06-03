import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

def main():
    csv_file = "datos-serial_T94.csv"
    
    print(f"Cargando datos desde {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_file}. Ejecuta primero el código C++.")
        sys.exit(1)

    # 1. Preparación de los datos
    steps = df['step'].unique()
    x_pos = df['x_pos'].unique()
    
    # Pivoteo para mapas de calor: filas=espacio (x), columnas=tiempo (step)
    # Transponemos para que el eje Y sea la posición y el eje X sea el tiempo
    T_grid = df.pivot(index='x_pos', columns='step', values='T').values
    dOH_grid = df.pivot(index='x_pos', columns='step', values='d_oh').values

    tiempo_segundos = steps * 1e-4  # step * DT (DT=1e-4)

    # =========================================================================
    # PARTE 1: GENERACIÓN DEL PANEL ESTÁTICO (PNG)
    # =========================================================================
    print("Generando panel de figuras estáticas...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Mapa de calor de la Temperatura T(x, t)
    im = axes[0].pcolormesh(tiempo_segundos, x_pos * 100, T_grid, cmap='inferno', shading='auto')
    axes[0].set_title('Mapa de Calor T(x, t)', fontsize=14)
    axes[0].set_xlabel('Tiempo [s]')
    axes[0].set_ylabel('Posición $x$ [cm]')
    axes[0].invert_yaxis() # Invertir para que la "piel" (x=0) esté arriba
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label('Temperatura [°C]')

    # Panel B: Perfiles espaciales en distintos instantes
    num_profiles = 5
    step_indices = np.linspace(0, len(steps)-1, num_profiles, dtype=int)
    for idx in step_indices:
        t_val = tiempo_segundos[idx]
        axes[1].plot(x_pos * 100, T_grid[:, idx], label=f't = {t_val:.2f} s')
    
    axes[1].set_title('Perfiles Térmicos', fontsize=14)
    axes[1].set_xlabel('Posición $x$ [cm]')
    axes[1].set_ylabel('Temperatura [°C]')
    axes[1].axvline(x=(10 * 1e-4)*100, color='gray', linestyle='--', label='Interfaz Piel/Bulk')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel C: Temperatura media vs Tiempo
    T_mean = np.mean(T_grid, axis=0)
    axes[2].plot(tiempo_segundos, T_mean, color='darkred', linewidth=2)
    axes[2].set_title('Relajación de la Temperatura Media', fontsize=14)
    axes[2].set_xlabel('Tiempo [s]')
    axes[2].set_ylabel(r'$\langle T \rangle$ [°C]')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mpemba_analisis.png", dpi=300)
    print("Panel guardado como 'mpemba_analisis.png'.")
    plt.close()


if __name__ == "__main__":
    main()
