import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import sys

def main():
    archivos = glob.glob("datos-cuda_T*.csv")
    
    if not archivos:
        print("Error: No se encontraron datos CSV de CUDA. Ejecuta run_sweep.sh primero.")
        sys.exit(1)

    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap('coolwarm')
    
    datos_simulacion = []
    for archivo in archivos:
        match = re.search(r'T(\d+)', archivo)
        if match:
            t_init = int(match.group(1))
            datos_simulacion.append((t_init, archivo))
            
    datos_simulacion.sort()
    
    for t_init, archivo in datos_simulacion:
        df = pd.read_csv(archivo)
        
        # Agrupación espacial para obtener la temperatura promedio del volumen
        df_mean = df.groupby('step')['T'].mean().reset_index()
        
        # CORRECCIÓN: Escala temporal ajustada al DT del kernel (DT = 1e-2)
        tiempo_segundos = df_mean['step'] * 1e-2
        temperatura_media = df_mean['T']
        
        # Normalizar color (0 a 1) usando la temperatura máxima del barrido (95C)
        color = cmap(t_init / 95.0)
        
        plt.plot(tiempo_segundos, temperatura_media, 
                 label=f'T_{{init}} = {t_init}°C', 
                 color=color, linewidth=2.5)

    # Estética de alta calidad para reporte científico
    plt.title('Efecto Mpemba Macroscópico: Crossover de Relajación Térmica', fontsize=15)
    plt.xlabel('Tiempo Físico [s]', fontsize=13)
    plt.ylabel(r'Temperatura Media $\langle T \rangle$ [°C]', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Estado Inicial', fontsize=11)
    
    # Línea de congelación
    plt.axhline(0, color='black', linewidth=1.5, linestyle=':')
    
    plt.tight_layout()
    nombre_salida = "mpemba_crossover_gpu.png"
    plt.savefig(nombre_salida, dpi=300)
    print(f"Gráfica de cruce generada exitosamente como '{nombre_salida}'.")

if __name__ == "__main__":
    main()
