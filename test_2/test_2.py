import cv2
import numpy as np
import psycopg2
import sys
import os

# Conexión a Postgres
try:
    conn = psycopg2.connect(
        dbname="facial_recognition",
        user="vlady",
        password="12345",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True
    cursor = conn.cursor()
    print("[INFO] Conectado a PostgreSQL exitosamente.")
except Exception as e:
    print("[ERROR] No se pudo conectar a la BD:", e)
    sys.exit(1)

# Precarga de los modelos
base_dir = "/home/vladythrash/Documentos/UV/Septimo Semestre/Embebidos/Facial_Recognition/models"
yunet_path = os.path.join(base_dir, "face_detection_yunet_2023mar.onnx")
sface_path = os.path.join(base_dir, "face_recognition_sface_2021dec.onnx")

detector = cv2.FaceDetectorYN.create(
    model=yunet_path, config="", input_size=(320, 320),
    score_threshold=0.9, nms_threshold=0.3, top_k=5000
)
recognizer = cv2.FaceRecognizerSF.create(model=sface_path, config="")

# Instancia del videocapture
cap = cv2.VideoCapture(0)
print("[INFO] Cámara iniciada. Presiona 'c' para escanear el rostro. 'q' para salir.")

UMBRAL_DISTANCIA = 0.637 # Equivalente a 0.363 de similitud SFace

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame) #Face detector

    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            cv2.rectangle(frame, box, (0, 255, 0), 2)

    cv2.imshow("Face ID - Sistema de Acceso", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and faces is not None:
        rostro = faces[0]
        aligned_face = recognizer.alignCrop(frame, rostro)
        descriptor = recognizer.feature(aligned_face) #Obtención de descriptores
        
        # Convertir array de numpy a string formato '[0.1, 0.2, ...]' para pgvector
        vec_str = str(descriptor[0].tolist())

        # Buscar el rostro más parecido en la BD
        cursor.execute("""
            SELECT p.id_usuario, p.nombre_completo, 
                   v.vector_caracteristicas <=> %s::vector AS distance 
            FROM logs v 
            JOIN person p ON v.id_usuario = p.id_usuario 
            ORDER BY distance LIMIT 1;
        """, (vec_str,))
        
        resultado = cursor.fetchone()

        # Validar si hay resultados y si superan el umbral
        if resultado and resultado[2] <= UMBRAL_DISTANCIA:
            id_usuario, nombre, distancia = resultado
            print(f"\n✅ ACCESO CONCEDIDO: Bienvenido, {nombre}")
            print(f"Distancia: {distancia:.4f} (Umbral max: {UMBRAL_DISTANCIA})")
            
            # Intentar insertar el nuevo vector (el trigger en SQL decidirá si lo guarda)
            cursor.execute("""
                INSERT INTO logs (id_usuario, vector_caracteristicas) 
                VALUES (%s, %s::vector);
            """, (id_usuario, vec_str))
            print("[INFO] Vector enviado a BD (El trigger evaluará su inserción).")

        else:
            print("\n❌ ROSTRO NO RECONOCIDO.")
            print("Por favor, mira la terminal para registrarte.")
            
            # Pausa la ventana de OpenCV para capturar texto en la terminal
            nuevo_nombre = input("Ingrese su nombre completo para registrarse: ")
            
            if nuevo_nombre.strip():
                # 1. Insertar en tabla person
                cursor.execute("""
                    INSERT INTO person (nombre_completo) VALUES (%s) RETURNING id_usuario;
                """, (nuevo_nombre,))
                nuevo_id = cursor.fetchone()[0]
                
                # 2. Insertar en tabla vector
                cursor.execute("""
                    INSERT INTO logs (id_usuario, vector_caracteristicas) 
                    VALUES (%s, %s::vector);
                """, (nuevo_id, vec_str))
                
                print(f"[EXITO] {nuevo_nombre} registrado correctamente con ID {nuevo_id}.\n")
            else:
                print("[INFO] Registro cancelado (nombre vacío).")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()