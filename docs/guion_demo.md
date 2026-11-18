# Guion de demo: cinco minutos sobre el prototipo de pantallas

Recorrido para la sustentación. Usa
[`etapa_6_despliegue/prototipo_pantallas/mesa-de-triaje.html`](../etapa_6_despliegue/prototipo_pantallas/mesa-de-triaje.html):
se descarga y se abre en el navegador, sin servidor ni instalación.

La página muestra una foto fija de una salida real del *pipeline*: 35,265
reclamos públicos de la CFPB recibidos del 4 al 17 de noviembre de 2024
(periodo de calibración), puntuados sin reentrenar. Cada paso del recorrido
enseña una de las seis hipótesis del informe con la cifra que la sostiene.
Las cifras de respaldo están en la tabla de hipótesis de la sección 5.1 y en
la sección 5.3 de `reports/trabajo_investigacion/trabajo.pdf`.

## Antes de empezar

- Abrir la página y dejarla en el menú principal.
- Leer en voz alta la franja superior: es una foto fija, no un sistema en
  vivo, y lo que se marca no se guarda.
- Tener a mano el PDF del informe en la tabla de hipótesis.

## Recorrido

| Minuto | Pantalla | Qué mostrar | Hipótesis y cifra del informe |
|---|---|---|---|
| 0:00 | Menú principal | Tres entradas: bandeja, observatorio y seguimiento. Ninguna pantalla decide | La decisión de triaje de la sección 1.1 |
| 0:30 | Bandeja de expedientes | Filtrar por un día. Cada fila trae el motivo más probable y su patrón | H6: el *pipeline* puntúa sin reentrenar; 12 minutos con GPU para dos semanas |
| 1:00 | Ficha de evidencia | Los tres motivos con su probabilidad. Pulsar «Confirmar motivo 1» y, en otro reclamo, «Guardar corrección» | H1: Macro-F1 0.271 frente a 0.076 de la regla por producto |
| 1:45 | Ficha de evidencia | Solución registrada y compensación, con su umbral. Decir que la respuesta fuera de plazo no se muestra porque no se anticipa del texto | H2: parcial; AP 0.570 y 0.270 en validación; la respuesta fuera de plazo quedó en 0.078 frente a 0.121 |
| 2:15 | Ficha de evidencia | Patrón, su fiabilidad, la marca de novedad y los cinco reclamos parecidos | H3: 40 patrones estables; cerca de 9 % de los reclamos cambia de patrón según el lote |
| 2:45 | Observatorio, por semana | Semana del 11/11/2024: observados, esperados, intervalo de 80 % y estado. Ordenado por diferencia | H4: 30 % menos error que la referencia |
| 3:15 | Patrón 14 | La alerta de crecimiento, el acumulado y la serie día a día con su banda | H4: la regla acumulada detecta 71 % de los crecimientos de 20 % con una falsa alarma al mes |
| 3:45 | Observatorio, por día | Cambiar a «Por día»: hay expectativa e intervalo, pero ninguna alerta | H5: parcial; el conteo diario sí (25 % menos error), la alerta diaria no |
| 4:15 | Seguimiento de alertas | Las dos alertas del periodo. Avanzar una hasta cerrarla con una lectura | Revisión humana obligatoria; el eco de las ráfagas es el límite declarado |
| 4:45 | Pie del observatorio | Señales de reentrenamiento: novedosos, cobertura del intervalo y Macro-F1 | H6: cobertura de 79 % en la prueba de punta a punta |

## Qué decir si preguntan

- **¿Los datos son reales?** Sí, son reclamos públicos de la CFPB de Estados
  Unidos y salidas reales de los modelos. No son datos de un banco peruano;
  trasladar el método a una entidad es la siguiente fase.
- **¿Está en producción?** No. Es una página estática con una foto fija. Lo
  que la persona confirma, corrige o marca no se guarda.
- **¿Por qué no aparece el texto del reclamo?** La salida del *pipeline* no
  lo copia. Cada identificador enlaza al registro público de la CFPB.
- **¿Detecta fraude?** No. Avisa cuando un patrón recibe más reclamos de lo
  esperado; la causa la determina la persona que revisa la alerta.
- **¿Cómo se actualiza?** Con un comando que toma cualquier carpeta de
  salida del *pipeline*:

  ```bash
  uv run python etapa_6_despliegue/prototipo_pantallas/construir_foto.py --score reports/pipeline/demo_2024-11-04
  ```

  La carpeta debe contener las tablas que escribe `python -m src.pipeline
  score`; en el repositorio solo queda su resumen `score.json`.

Guion redactado con ayuda de IA (Claude); las cifras se tomaron del informe y
deben repasarse contra él antes de la sustentación.
