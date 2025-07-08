import pandas as pd
import numpy as np
from scipy.stats import t

# Leer el archivo de datos
df = pd.read_csv('velocidad_internet_ucu.txt')
total_mediciones_original = len(df)  # Guardar longitud original

print("="*80)
print("PASO 1: Filtrar observaciones para Central y Semprún")
print("="*80)
edificios = ['Central', 'Semprún']
df_filtrado = df[df['Edificio'].isin(edificios)]

# Guardar datos filtrados en archivo TXT
with open("datos_filtrados.txt", "w", encoding="utf-8") as f:
    f.write("Datos completos filtrados para los edificios Central y Semprún\n")
    f.write("="*80 + "\n")
    f.write(f"Total de mediciones: {len(df_filtrado)}\n")
    f.write(f"Mediciones para Central: {len(df_filtrado[df_filtrado['Edificio'] == 'Central'])}\n")
    f.write(f"Mediciones para Semprún: {len(df_filtrado[df_filtrado['Edificio'] == 'Semprún'])}\n\n")
    f.write(df_filtrado.to_string(index=False))

print(f"Total de mediciones filtradas: {len(df_filtrado)}")
print(f"Mediciones para Central: {len(df_filtrado[df_filtrado['Edificio'] == 'Central'])}")
print(f"Mediciones para Semprún: {len(df_filtrado[df_filtrado['Edificio'] == 'Semprún'])}")
print("Datos filtrados guardados en 'datos_filtrados.txt'")
print("\n" + "="*80 + "\n")

print("="*80)
print("PASO 2: Calcular estadísticas descriptivas para cada edificio")
print("="*80)
central = df_filtrado[df_filtrado['Edificio'] == 'Central']
semprun = df_filtrado[df_filtrado['Edificio'] == 'Semprún']

n1 = len(central)
x1 = central['Velocidad Mb/s'].mean()
s1 = central['Velocidad Mb/s'].std()

n2 = len(semprun)
x2 = semprun['Velocidad Mb/s'].mean()
s2 = semprun['Velocidad Mb/s'].std()

print(f"Central: n = {n1}, Media = {x1:.2f} Mbps, Desviación estándar = {s1:.2f}")
print(f"Semprún: n = {n2}, Media = {x2:.2f} Mbps, Desviación estándar = {s2:.2f}")
print("\n" + "="*80 + "\n")

print("="*80)
print("PASO 3: Calcular estadístico t")
print("="*80)
# Calcular estadístico t
numerador = x1 - x2
denominador = np.sqrt(s1**2/n1 + s2**2/n2)
t_stat = numerador / denominador

# Calcular grados de libertad (Welch-Satterthwaite)
num_df = (s1**2/n1 + s2**2/n2)**2
den_df = (s1**4)/(n1**2*(n1-1)) + (s2**4)/(n2**2*(n2-1))
df = num_df / den_df

print(f"Fórmula: t = (X̄1 - X̄2) / √(s1²/n1 + s2²/n2)")
print(f"         = ({x1:.2f} - {x2:.2f}) / √({s1**2/n1:.2f} + {s2**2/n2:.2f})")
print(f"         = {numerador:.2f} / √{s1**2/n1 + s2**2/n2:.2f}")
print(f"         = {numerador:.2f} / {denominador:.2f}")
print(f"         = {t_stat:.4f}")
print(f"\nGrados de libertad (Welch-Satterthwaite): {df:.2f}")
print("\n" + "="*80 + "\n")

print("="*80)
print("PASO 4: Calcular p-valor")
print("="*80)
# Calcular p-valor (prueba unilateral izquierda)
p_valor = t.cdf(t_stat, df)

print(f"Prueba unilateral izquierda (H1: μ_Central < μ_Semprún)")
print(f"p-valor = P(T ≤ {t_stat:.4f}) con {df:.2f} grados de libertad")
print(f"p-valor = {p_valor:.15f}")
print("\n" + "="*80 + "\n")

print("="*80)
print("PASO 5: Determinar si se rechaza la hipótesis nula (α = 0.05)")
print("="*80)
alpha = 0.05
rechazo = p_valor < alpha

print(f"α = {alpha}")
print(f"p-valor = {p_valor:.15f}")
print(f"Decisión: {'Rechazamos H0' if rechazo else 'No rechazamos H0'} ya que p-valor {'<' if rechazo else '>='} α")
print("\n" + "="*80 + "\n")

print("="*80)
print("PASO 6: Interpretar los resultados")
print("="*80)
interpretacion = (
    "Existe evidencia estadísticamente significativa (p < 0.001) para afirmar que "
    "la velocidad promedio de internet en el edificio Central es significativamente "
    "menor que en el edificio Semprún. "
    "Esta diferencia podría tener implicancias importantes en el uso académico del internet, "
    "especialmente para actividades que requieren alto ancho de banda como videoconferencias, "
    "transmisión de videos educativos o descarga de materiales pesados. "
    "Se recomienda investigar las causas de esta diferencia y considerar mejoras "
    "en la infraestructura de red del edificio Central."
)
print(interpretacion)

# Guardar resultados del análisis en archivo
with open("resultados_analisis.txt", "w", encoding="utf-8") as f:
    f.write("RESULTADOS COMPLETOS DEL ANÁLISIS ESTADÍSTICO\n")
    f.write("="*60 + "\n\n")
    
    f.write("PASO 1: Filtrado de datos\n")
    f.write(f"- Total de mediciones en el conjunto original: {total_mediciones_original}\n")
    f.write(f"- Mediciones filtradas (Central y Semprún): {len(df_filtrado)}\n")
    f.write(f"  - Central: {n1} mediciones\n")
    f.write(f"  - Semprún: {n2} mediciones\n")
    f.write(f"- Los datos completos filtrados se encuentran en 'datos_filtrados.txt'\n\n")
    
    f.write("PASO 2: Estadísticas descriptivas\n")
    f.write(f"Central: n = {n1}, Media = {x1:.2f} Mbps, DE = {s1:.2f}\n")
    f.write(f"Semprún: n = {n2}, Media = {x2:.2f} Mbps, DE = {s2:.2f}\n\n")
    
    f.write("PASO 3: Cálculo del estadístico t\n")
    f.write(f"t = (X̄1 - X̄2) / √(s1²/n1 + s2²/n2)\n")
    f.write(f"  = ({x1:.2f} - {x2:.2f}) / √({s1**2/n1:.2f} + {s2**2/n2:.2f})\n")
    f.write(f"  = {t_stat:.4f}\n")
    f.write(f"Grados de libertad (Welch-Satterthwaite): {df:.2f}\n\n")
    
    f.write("PASO 4: Cálculo del p-valor\n")
    f.write(f"p-valor (prueba unilateral izquierda) = {p_valor:.15f}\n\n")
    
    f.write("PASO 5: Decisión estadística (α = 0.05)\n")
    f.write(f"p-valor = {p_valor:.15f} {'<' if rechazo else '>='} α = {alpha}\n")
    f.write(f"Decisión: {'Rechazamos H0' if rechazo else 'No rechazamos H0'}\n\n")
    
    f.write("PASO 6: Interpretación de resultados\n")
    f.write(interpretacion)

print("\n\nResultados del análisis guardados en 'resultados_analisis.txt'")