# data/

Los datos **no se versionan** (ADR-0002): son datos con licencia de FC Barcelona / Metrica Sports y pesan mucho (vídeo).

Cada persona descarga los datos oficiales y los coloca aquí con esta estructura (se ajustará cuando veamos el paquete real):

```
data/
  raw/          # tal cual los da la organización, sin tocar
  interim/      # convertidos a formato común (p. ej. parquet en coordenadas de campo)
  labels/       # etiquetas manuales nuestras (si las hacemos) -> estas SÍ se comparten por Drive
```

Fuente de los datos y fecha de descarga: _pendiente_. Cuando los tengamos, documentad aquí el inventario (ficheros, tamaño, formato y fps).
