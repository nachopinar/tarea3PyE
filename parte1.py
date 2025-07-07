import pandas as pd
import numpy as np
from scipy.stats import chi2

# 1. Leer el archivo CSV
data = pd.read_csv('muestra_ech.csv')

# Manejar valores faltantes
data = data.dropna(subset=['ingreso'])
data = data[data['ingreso'] != '']
data['ingreso'] = data['ingreso'].astype(float)

# ===================================================================
# Punto 1: Calcular ingreso per cápita de cada hogar
# ===================================================================
data['ingreso_per_capita'] = data['ingreso'] / data['personas_hogar']

# Mostrar primeros 10 casos en terminal
print("\n=== PUNTO 1: Primeros 10 hogares con ingreso per cápita ===")
print(data[['hogar', 'ingreso', 'personas_hogar', 'ingreso_per_capita']].head(10))

# Guardar resultados en TXT
with open('resultado_punto1.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADO PUNTO 1 ===\n")
    f.write("Ingreso per cápita calculado para cada hogar:\n\n")
    f.write("Hogar | Ingreso total | Personas | Ingreso per cápita\n")
    f.write("----------------------------------------------------\n")
    
    for index, row in data.iterrows():
        f.write(f"{row['hogar']} | {row['ingreso']:.2f} | {row['personas_hogar']} | {row['ingreso_per_capita']:.2f}\n")
    
    f.write("\n----------------------------------------------------\n")
    f.write(f"Total de hogares procesados: {len(data)}\n")
    f.write("Archivo generado: resultado_punto1.txt")

# ===================================================================
# Punto 2: Clasificar en quintiles usando percentiles
# ===================================================================
percentiles = np.percentile(data['ingreso_per_capita'], [20, 40, 60, 80])
data['quintil'] = 1
data.loc[data['ingreso_per_capita'] > percentiles[0], 'quintil'] = 2
data.loc[data['ingreso_per_capita'] > percentiles[1], 'quintil'] = 3
data.loc[data['ingreso_per_capita'] > percentiles[2], 'quintil'] = 4
data.loc[data['ingreso_per_capita'] > percentiles[3], 'quintil'] = 5

# Mostrar primeros 10 casos en terminal
print("\n=== PUNTO 2: Primeros 10 hogares con quintil asignado ===")
print(data[['hogar', 'ingreso_per_capita', 'quintil']].head(10))

# Guardar resultados en TXT
with open('resultado_punto2.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADO PUNTO 2 ===\n")
    f.write("Clasificación de hogares en quintiles según ingreso per cápita\n\n")
    f.write(f"Percentiles calculados:\n")
    f.write(f"  - 20%: {percentiles[0]:.2f}\n")
    f.write(f"  - 40%: {percentiles[1]:.2f}\n")
    f.write(f"  - 60%: {percentiles[2]:.2f}\n")
    f.write(f"  - 80%: {percentiles[3]:.2f}\n\n")
    
    f.write("Distribución de hogares por quintil:\n")
    quintil_counts = data['quintil'].value_counts().sort_index()
    for quintil, count in quintil_counts.items():
        f.write(f"Quintil {quintil}: {count} hogares ({count/len(data)*100:.1f}%)\n")
    
    f.write("\nDetalle completo de clasificación:\n")
    f.write("Hogar | Ingreso per cápita | Quintil\n")
    f.write("------------------------------------\n")
    
    for index, row in data.iterrows():
        f.write(f"{row['hogar']} | {row['ingreso_per_capita']:.2f} | {row['quintil']}\n")
    
    f.write("\n------------------------------------\n")
    f.write(f"Total de hogares clasificados: {len(data)}\n")
    f.write("Archivo generado: resultado_punto2.txt")

# ===================================================================
# Punto 3: Filtrar quintil superior
# ===================================================================
quintil_superior = data[data['quintil'] == 5]

# Mostrar primeros 10 casos en terminal
print("\n=== PUNTO 3: Primeros 10 hogares del quintil superior ===")
print(quintil_superior[['hogar', 'ingreso', 'personas_hogar', 'ingreso_per_capita', 'departamento']].head(10))

# Guardar resultados en TXT
with open('resultado_punto3.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADO PUNTO 3 ===\n")
    f.write("Hogares del quintil superior (20% con mayor ingreso per cápita)\n\n")
    f.write(f"Total de hogares en quintil superior: {len(quintil_superior)}\n")
    f.write(f"Porcentaje respecto al total: {len(quintil_superior)/len(data)*100:.1f}%\n\n")
    
    f.write("Detalle completo de hogares en quintil superior:\n")
    f.write("Hogar | Ingreso total | Personas | Ingreso per cápita | Departamento\n")
    f.write("-------------------------------------------------------------------\n")
    
    for index, row in quintil_superior.iterrows():
        f.write(f"{row['hogar']} | {row['ingreso']:.2f} | {row['personas_hogar']} | {row['ingreso_per_capita']:.2f} | {row['departamento']}\n")
    
    f.write("\n-------------------------------------------------------------------\n")
    f.write(f"Total de hogares en quintil superior: {len(quintil_superior)}\n")
    f.write("Archivo generado: resultado_punto3.txt")

# ===================================================================
# Puntos 4-8 (se mantienen similares pero con salida en consola)
# ===================================================================
# 4. Tabla de frecuencias observadas por departamento
frec_observada = quintil_superior['departamento'].value_counts().sort_index()
todos_departamentos = pd.Series(index=range(1, 20), dtype=float).fillna(0)
frec_observada = todos_departamentos.add(frec_observada, fill_value=0).astype(int)

# 5. Calcular frecuencias esperadas
total_observado = len(quintil_superior)
num_departamentos = 19
frec_esperada = total_observado / num_departamentos

# 6. Calcular estadístico chi-cuadrado
chi_cuadrado = 0
for i in range(1, 20):
    observado = frec_observada[i]
    chi_cuadrado += (observado - frec_esperada) ** 2 / frec_esperada

# 7. Determinar rechazo de hipótesis
gl = num_departamentos - 1
valor_critico = chi2.ppf(0.95, gl)
rechazar_h0 = chi_cuadrado > valor_critico

# Resultados en consola para puntos 4-8
print("\n=== RESULTADOS FINALES ===")
print("\n--- Punto 4: Frecuencias observadas por departamento ---")
print(frec_observada)

print("\n--- Punto 5: Frecuencia esperada ---")
print(f"Total hogares quintil superior: {total_observado}")
print(f"Número de departamentos: {num_departamentos}")
print(f"Frecuencia esperada: {frec_esperada:.2f}")

print("\n--- Punto 6: Estadístico chi-cuadrado ---")
print(f"χ² = {chi_cuadrado:.4f}")

print("\n--- Punto 7: Test de hipótesis ---")
print(f"Grados de libertad: {gl}")
print(f"Valor crítico (α=0.05): {valor_critico:.4f}")
print(f"¿Se rechaza H0?: {'Sí' if rechazar_h0 else 'No'}")

print("\n--- Punto 8: Interpretación ---")
if rechazar_h0:
    print("Se rechaza la hipótesis nula. Existen diferencias significativas")
    print("en la distribución de hogares de altos ingresos entre departamentos.")
    print("Esto sugiere que la riqueza no está distribuida uniformemente en el país,")
    print("sino que hay departamentos con mayor concentración de hogares de altos ingresos.")
else:
    print("No se rechaza la hipótesis nula. La distribución de hogares")
    print("de altos ingresos es consistente con una distribución uniforme entre departamentos.")
    print("Esto sugiere que la riqueza está distribuida de manera equilibrada en todo el país.")

# Guardar resultados finales en TXT
with open('resultados_finales.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADOS FINALES (PUNTOS 4-8) ===\n\n")
    
    f.write("--- Punto 4: Frecuencias observadas por departamento ---\n")
    f.write(str(frec_observada) + "\n\n")
    
    f.write("--- Punto 5: Frecuencia esperada ---\n")
    f.write(f"Total hogares quintil superior: {total_observado}\n")
    f.write(f"Número de departamentos: {num_departamentos}\n")
    f.write(f"Frecuencia esperada: {frec_esperada:.2f}\n\n")
    
    f.write("--- Punto 6: Estadístico chi-cuadrado ---\n")
    f.write(f"χ² = {chi_cuadrado:.4f}\n\n")
    
    f.write("--- Punto 7: Test de hipótesis ---\n")
    f.write(f"Grados de libertad: {gl}\n")
    f.write(f"Valor crítico (α=0.05): {valor_critico:.4f}\n")
    f.write(f"¿Se rechaza H0?: {'Sí' if rechazar_h0 else 'No'}\n\n")
    
    f.write("--- Punto 8: Interpretación ---\n")
    if rechazar_h0:
        f.write("Se rechaza la hipótesis nula. Existen diferencias significativas\n")
        f.write("en la distribución de hogares de altos ingresos entre departamentos.\n")
        f.write("Esto sugiere que la riqueza no está distribuida uniformemente en el país,\n")
        f.write("sino que hay departamentos con mayor concentración de hogares de altos ingresos.\n")
    else:
        f.write("No se rechaza la hipótesis nula. La distribución de hogares\n")
        f.write("de altos ingresos es consistente con una distribución uniforme entre departamentos.\n")
        f.write("Esto sugiere que la riqueza está distribuida de manera equilibrada en todo el país.\n")

print("\nArchivos generados:")
print("- resultado_punto1.txt (Punto 1: Ingreso per cápita)")
print("- resultado_punto2.txt (Punto 2: Clasificación por quintiles)")
print("- resultado_punto3.txt (Punto 3: Quintil superior)")
print("- resultados_finales.txt (Puntos 4-8: Resultados finales)")