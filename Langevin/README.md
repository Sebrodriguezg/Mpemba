# Escalado de Temperaturas para la Simulación

Las temperaturas se expresan como una razón respecto a la temperatura del baño térmico en Kelvin.

## 1. Temperatura de referencia

La temperatura del baño se toma como referencia:

\[
T_{bath} = 1.0
\]

## 2. Temperatura ambiente

Si la temperatura ambiente es de \(18^\circ C\) y la temperatura del baño es de \(-12^\circ C\), entonces:

\[
T_{warm\_sim}
=
\frac{18 + 273.15}{-12 + 273.15}
\approx 1.114876508
\]

## 3. Temperatura caliente

Para una temperatura caliente de \(75^\circ C\):

\[
T_{hot\_sim}
=
\frac{75 + 273.15}{-12 + 273.15}
\approx 1.333141872 
\]

## Resumen

| Temperatura física | Temperatura escalada |
|-------------------|----------------------|
| \(T_{bath}\) (-12 °C) | 1.00 |
| \(T_{warm}\) (18 °C) | 1.15 |
| \(T_{hot}\) (75 °C) | 1.38 |

## Fórmula general

Para convertir una temperatura física \(T\) (en °C) a la temperatura adimensional utilizada en la simulación:

\[
T_{sim}
=
\frac{T + 273.15}{T_{bath} + 273.15}
\]

donde \(T_{bath}\) es la temperatura del baño térmico expresada en grados Celsius.
