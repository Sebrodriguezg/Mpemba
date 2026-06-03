import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    # Permite pasar el nombre del archivo por consola para no tener que editar el código
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "datos-cuda_T95.csv" # Archivo por defecto
    
    print(f"Cargando datos desde {csv_file}...")
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró el archivo '{csv_file}'.")
        print("Uso: python3 plot_mpemba.py <nombre_del_archivo.csv>")
        sys.exit(1)
        
    df = pd.read_csv(csv_file)

    # 1. Preparación de los datos
    steps = df['step'].unique()
    x_pos = df['x_pos'].unique()
    
    # Pivoteo para mapas de calor: filas=espacio (x), columnas=tiempo (step)
    T_grid = df.pivot(index='x_pos', columns='step', values='T').values
    dOH_grid = df.pivot(index='x_pos', columns='step', values='d_oh').values

    # CORRECCIÓN DE ESCALA TEMPORAL Y ESPACIAL: 
    # En tu CUDA LBM, DT = 1e-2 y DX = 1e-4
    tiempo_segundos = steps * 1e-2  
    x_cm = x_pos * 100 

    # =========================================================================
    # GENERACIÓN DEL PANEL ESTÁTICO 2x2 (PNG)
    # =========================================================================
    print("Generando panel de figuras estáticas 2x2...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ---------------------------------------------------------
    # Panel A: Mapa de calor de la Temperatura T(x, t)
    # ---------------------------------------------------------
    im = ax1.pcolormesh(tiempo_segundos, x_cm, T_grid, cmap='inferno', shading='auto')
    ax1.set_title('A. Evolución Espacio-Temporal: $T(x, t)$', fontsize=14, weight='bold')
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Posición $x$ [cm]')
    ax1.invert_yaxis() # Invertir para que la "piel" (x=0) esté arriba
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Temperatura [°C]')

    # ---------------------------------------------------------
    # Panel B: Perfiles espaciales térmicos
    # ---------------------------------------------------------
    num_profiles = 6
    step_indices = np.linspace(0, len(steps)-1, num_profiles, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_profiles))
    
    for c, idx in zip(colors, step_indices):
        t_val = tiempo_segundos[idx]
        ax2.plot(x_cm, T_grid[:, idx], label=f't = {t_val:.1f} s', color=c, lw=2)
    
    ax2.set_title('B. Perfiles Térmicos $T(x)$', fontsize=14, weight='bold')
    ax2.set_xlabel('Posición $x$ [cm]')
    ax2.set_ylabel('Temperatura [°C]')
    
    # Interfaz Piel/Bulk (SKIN_NODES = 50, DX = 1e-4m -> 0.5 cm)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Interfaz Piel/Bulk')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.7)

    # ---------------------------------------------------------
    # Panel C: Temperatura media vs Tiempo
    # ---------------------------------------------------------
    T_mean = np.mean(T_grid, axis=0)
    ax3.plot(tiempo_segundos, T_mean, color='darkred', linewidth=2.5)
    ax3.set_title('C. Relajación de la Temperatura Media', fontsize=14, weight='bold')
    ax3.set_xlabel('Tiempo [s]')
    ax3.set_ylabel(r'$\langle T \rangle$ Global [°C]')
    ax3.axhline(0, color='black', linestyle='-.', lw=1, alpha=0.7) # Línea de congelación
    ax3.grid(True, linestyle=':', alpha=0.7)

    # ---------------------------------------------------------
    # Panel D: Memoria Estructural (Enlace O:H-O) de Zhang
    # ---------------------------------------------------------
    dOH_mean = np.mean(dOH_grid, axis=0)
    ax4.plot(tiempo_segundos, dOH_mean, color='teal', linewidth=2.5, label=r'Longitud media $\langle d_{OH} \rangle$')
    ax4.set_title('D. Relajación Estructural del Enlace Covalente', fontsize=14, weight='bold')
    ax4.set_xlabel('Tiempo [s]')
    ax4.set_ylabel(r'Longitud de Enlace $\langle d_{OH} \rangle$ [$\AA$]')
    
    # Anotaciones físicas
    ax4.annotate('Alta energía almacenada\n(Enlace comprimido)', 
                 xy=(tiempo_segundos[0], dOH_mean[0]), 
                 xytext=(tiempo_segundos[len(tiempo_segundos)//10], dOH_mean[0] + 0.0001),
                 arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5), fontsize=10)
                 
    ax4.legend(loc='lower right')
    ax4.grid(True, linestyle=':', alpha=0.7)

    # Ajustes finales y guardado
    plt.tight_layout()
    # Genera un nombre de salida basado en el archivo de entrada
    nombre_salida = f"analisis_{os.path.splitext(os.path.basename(csv_file))[0]}.png"
    plt.savefig(nombre_salida, dpi=300, bbox_inches='tight')
    print(f"Panel analítico guardado exitosamente como '{nombre_salida}'.")
    plt.close()

if __name__ == "__main__":
    main()
