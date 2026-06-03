#!/bin/bash

# Salir inmediatamente si un comando falla
set -e

# 1. Compilar el código C++
echo "Compilando serial.cpp..."
g++ -std=c++17 -O3 serial.cpp -o mpemba_serial
echo "Compilación exitosa."

# 2. Definir las temperaturas iniciales para el barrido
TEMPERATURES=(20 40 60 80 94)

# 3. Ejecutar la simulación para cada temperatura
echo "Iniciando barrido paramétrico..."
for T in "${TEMPERATURES[@]}"; do
    echo "Ejecutando simulación para temperatura inicial: ${T}°C"
    ./mpemba_serial $T
done

echo "Todas las simulaciones han terminado."

# 4. Ejecutar el script de Python para unificar y graficar
echo "Generando gráfica comparativa..."
python3 plot_sweep.py


# 5. Ejecutar el script de Python para analizar T=94
echo "Generando gráficas detalladas de T 94..."
python3 plot_mpemba.py

echo "Proceso completo."
