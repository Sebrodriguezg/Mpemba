import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def main():
    # Encontrar todos los archivos CSV generados por el barrido
    archivos = glob.glob("datos-serial_T*.csv")
    
    if not archivos:
        print("No se encontraron archivos CSV. Ejecuta el script de bash primero.")
        return

    plt.figure(figsize=(10, 7))
    
    # Colores cálidos para temperaturas altas, fríos para bajas
    cmap = plt.get_cmap('coolwarm')
    
    # Extraer las temperaturas de los nombres de archivo y ordenarlas
    datos_simulacion = []
    for archivo in archivos:
        # Extraer el número del nombre del archivo (ej. datos-serial_T40.csv -> 40)
        match = re.search(r'T(\d+)', archivo)
        if match:
            t_init = int(match.group(1))
            datos_simulacion.append((t_init, archivo))
            
    # Ordenar de menor a mayor temperatura para que los colores tengan sentido
    datos_simulacion.sort()
    
    # Graficar la temperatura media de cada ejecución
    for t_init, archivo in datos_simulacion:
        df = pd.read_csv(archivo)
        
        # Agrupar por el paso de tiempo (step) y calcular el promedio espacial
        df_mean = df.groupby('step')['T'].mean().reset_index()
        
        # Convertir 'step' a segundos (asumiendo DT = 1e-4 del C++)
        tiempo_segundos = df_mean['step'] * 1e-4
        temperatura_media = df_mean['T']
        
        # Normalizar el color en base a la temperatura máxima (94)
        color = cmap(t_init / 94.0)
        
        plt.plot(tiempo_segundos, temperatura_media, 
                 label=f'T inicial = {t_init}°C', 
                 color=color, linewidth=2.5)

    # Configuraciones estéticas de la gráfica
    plt.title('Relajación Térmica y Búsqueda del Efecto Mpemba (LBM D1Q3)', fontsize=15)
    plt.xlabel('Tiempo de simulación [s]', fontsize=12)
    plt.ylabel(r'Temperatura Media $\langle T \rangle$ [°C]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Condiciones', fontsize=10)
    
    # Línea horizontal indicando el 0°C (inicio de congelación)
    plt.axhline(0, color='black', linewidth=1.5, linestyle=':')
    
    plt.tight_layout()
    nombre_salida = "mpemba_crossover.png"
    plt.savefig(nombre_salida, dpi=300)
    print(f"Gráfica comparativa guardada exitosamente como '{nombre_salida}'.")

if __name__ == "__main__":
    main()
