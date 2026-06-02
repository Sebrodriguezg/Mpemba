# Escalado de Temperaturas para la Simulación

Las temperaturas utilizadas en la simulación se expresan como una razón respecto a la temperatura del baño térmico. Para ello, todas las temperaturas se convierten primero a Kelvin y luego se normalizan usando la temperatura del baño como referencia.

## 1. Temperatura de referencia

La temperatura del baño se toma como referencia:

```text
T_bath = 1.0
```

## 2. Temperatura ambiente

Si la temperatura ambiente es de **18 °C** y la temperatura del baño es de **−12 °C**, entonces:

```text
T_warm_sim = (18 + 273.15) / (-12 + 273.15)
           ≈ 1.114876508
```

## 3. Temperatura caliente

Para una temperatura caliente de **75 °C**:

```text
T_hot_sim = (75 + 273.15) / (-12 + 273.15)
          ≈ 1.333141872
```

## Resumen

| Temperatura física | Temperatura escalada |
|-------------------|----------------------|
| T_bath (-12 °C) | 1.0000 |
| T_warm (18 °C) | 1.1149 |
| T_hot (75 °C) | 1.3331 |

## Fórmula general

Para cualquier temperatura física \(T\) expresada en grados Celsius:

```text
T_sim = (T + 273.15) / (T_bath + 273.15)
```

donde:

- `T` es la temperatura física en °C.
- `T_bath` es la temperatura del baño en °C.
- `T_sim` es la temperatura adimensional utilizada en la simulación.

## Ejemplo

Suponiendo un baño térmico a −12 °C:

```text
T_bath = -12 °C
```

entonces:

```text
T_sim(18 °C) = (18 + 273.15) / (-12 + 273.15)
             ≈ 1.1149

T_sim(75 °C) = (75 + 273.15) / (-12 + 273.15)
             ≈ 1.3331
```

Estos valores son los que deben emplearse directamente en la simulación para representar las temperaturas física ambiente y caliente manteniendo la escala relativa respecto al baño térmico.