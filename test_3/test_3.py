# Ahora el sistema combina el roconocimento y consulta de un rostro con un modelo en Ollama.
# 1.- Si se detecta un rostro se buscan registros similares en la base de datos.
# 2.- Si el rostro existe, se enciende el modelo de Ollama y se inyecta el contexo de la ultima charla (de momento solo el nombre).
# 3.- Si no existe un rostro, primero se pregunta el nombre, se hace el registro y finalmente se enciende el modelo.
# 4.- El modelo se queda en ejecución hasta que el usuario decide terminar la charla.

import cv2
import numpy as np
import psycopg2
import sys
import os
import ollama  # Importación de Ollama

def iniciar_chat(nombre, es_nuevo):
    """
    Inyecta el contexto inicial en Llama 3.1 y maneja el bucle de texto en consola.
    """
    if es_nuevo:
        msg_usuario = {
            "role": "user",
            "content": f"[LOG BIOMÉTRICO]: Nuevo rostro detectado. El humano se llama {nombre}."
        }
    else:
        msg_usuario = {
            "role": "user",
            "content": f"[LOG BIOMÉTRICO]: Rostro conocido detectado. La entidad biológica {nombre} ha regresado. El ya estuvo en el evento antes."
        }

    # Solo enviamos el mensaje del usuario; el Modelfile pone el contexto de sistema
    historial = [msg_usuario]
    
    print("\n" + "="*50)
    print("Despertando a Aporía...")
    
    # El modelo da el primer saludo basado en el contexto inyectado
    respuesta = ollama.chat(model='alienado_llama:latest', messages=historial)
    msg_llm = respuesta['message']['content']
    
    print(f"\nAsistente: {msg_llm}")
    historial.append({"role": "assistant", "content": msg_llm})
    
    # Bucle de chat en terminal
    while True:
        user_input = input(f"\nUser {nombre}: ")
        
        if user_input.lower() in ['salir', 'adios', 'q']:
            print("\nFinalizando chat. Volviendo al modo de escaneo facial...")
            print("="*50 + "\n")
            break
            
        historial.append({"role": "user", "content": user_input})
        
        respuesta = ollama.chat(model='alienado_llama:latest ', messages=historial)
        msg_llm = respuesta['message']['content']
        
        print(f"Asistente: {msg_llm}")
        historial.append({"role": "assistant", "content": msg_llm})


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

cap = cv2.VideoCapture(0)
print("[INFO] Cámara iniciada. Presiona 'c' para escanear el rostro. 'q' para salir.")

UMBRAL_DISTANCIA = 0.637

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            cv2.rectangle(frame, box, (0, 255, 0), 2)

    cv2.imshow("Face ID - Sistema de Acceso", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and faces is not None:
        rostro = faces[0]
        aligned_face = recognizer.alignCrop(frame, rostro)
        descriptor = recognizer.feature(aligned_face)
        
        vec_str = str(descriptor[0].tolist())

        cursor.execute("""
            SELECT p.id_usuario, p.nombre_completo, 
                   v.vector_caracteristicas <=> %s::vector AS distance 
            FROM logs v 
            JOIN person p ON v.id_usuario = p.id_usuario 
            ORDER BY distance LIMIT 1;
        """, (vec_str,))
        
        resultado = cursor.fetchone()

        if resultado and resultado[2] <= UMBRAL_DISTANCIA:
            id_usuario, nombre, distancia = resultado
            print(f"\n✅ ACCESO CONCEDIDO: Bienvenido, {nombre}")
            print(f"Distancia: {distancia:.4f} (Umbral max: {UMBRAL_DISTANCIA})")
            
            cursor.execute("""
                INSERT INTO logs (id_usuario, vector_caracteristicas) 
                VALUES (%s, %s::vector);
            """, (id_usuario, vec_str))
            print("[INFO] Vector enviado a BD (El trigger evaluará su inserción).")
            
            # --- SE LLAMA AL LLM PARA USUARIO CONOCIDO ---
            iniciar_chat(nombre, es_nuevo=False)

        else:
            print("\n❌ ROSTRO NO RECONOCIDO.")
            print("Por favor, mira la terminal para registrarte.")
            
            nuevo_nombre = input("Ingrese su nombre completo para registrarse: ")
            
            if nuevo_nombre.strip():
                cursor.execute("""
                    INSERT INTO person (nombre_completo) VALUES (%s) RETURNING id_usuario;
                """, (nuevo_nombre,))
                nuevo_id = cursor.fetchone()[0]
                
                cursor.execute("""
                    INSERT INTO logs (id_usuario, vector_caracteristicas) 
                    VALUES (%s, %s::vector);
                """, (nuevo_id, vec_str))
                
                print(f"[EXITO] {nuevo_nombre} registrado correctamente con ID {nuevo_id}.\n")
                
                # --- SE LLAMA AL LLM PARA USUARIO NUEVO ---
                iniciar_chat(nuevo_nombre, es_nuevo=True)
            else:
                print("[INFO] Registro cancelado (nombre vacío).")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()