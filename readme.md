# Reconocimiento facial: Toma y consulta de descriptores unicos (ID'S).

El siguiente pipeline genera a partir de descriptores faciales valores, claves unicas o ID's a fin de obtener un sistema de reconocimiento. Cada rostro puede generar descriptores distintos, por lo que cada persona estara asociada a un ID o clave generado a partir de las caracteristicas de su rostro, esto con la finalidad de generar un sistema "inteligente" que puede recordar con que personas a interactuado anteriormente.

La arquitectura propuesta es la siguiente:
- Captura de video-imagenes mediante la webcam.
- Procesamiento y obtención de descriptores faciales.
- Generación de ID y consulta en BD.
- Gráficación del resultado y sistema de interacción (agregación de un nuevo registro).