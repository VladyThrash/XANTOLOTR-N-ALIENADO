import cv2
import numpy as np
import sys

#Cargar los modelos en memoria antes del bucle del videocapture.
try:
    detector = cv2.FaceDetectorYN.create(
        model="/home/vladythrash/Documentos/UV/Septimo Semestre/Embebidos/Facial_Recognition/models/face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(320, 320), # Este tamaño se actualizará dinámicamente en el bucle
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        model="/home/vladythrash/Documentos/UV/Septimo Semestre/Embebidos/Facial_Recognition/models/face_recognition_sface_2021dec.onnx",
        config=""
    )
except cv2.error as e:
    print("Error al cargar los modelos. Asegúrate de que los archivos .onnx estén en este directorio.")
    sys.exit(1)

#Iniciar la captura de video (webcam).
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    sys.exit(1)

print("Cámara iniciada.")
print("[INFO] Presiona 'c' para CAPTURAR el descriptor del rostro.")
print("[INFO] Presiona 'q' para SALIR.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    # Actualizar dinámicamente las dimensiones de entrada del detector
    height, width, _ = frame.shape
    detector.setInputSize((width, height))

    # Detectar rostros en el frame actual
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Extraer las coordenadas del Bounding Box (x, y, ancho, alto)
            box = list(map(int, face[:4]))
            
            # Dibujar el rectángulo verde en el rostro detectado
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            
            # Dibujar los 5 landmarks faciales (ojos, nariz, comisuras)
            landmarks = list(map(int, face[4:14]))
            for i in range(5):
                cv2.circle(frame, (landmarks[i*2], landmarks[i*2+1]), 2, (0, 0, 255), -1)

    # Mostrar la ventana de video
    cv2.imshow("Face ID - Modo Captura", frame)

    # Capturar eventos del teclado
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Extraer el descriptor si el usuario presiona 'c' y hay un rostro en pantalla
        if faces is not None:
            # Tomamos el primer rostro detectado (faces[0])
            rostro_principal = faces[0]
            
            # Alinear y recortar usando los landmarks internos de YuNet
            aligned_face = recognizer.alignCrop(frame, rostro_principal)
            
            # Extraer el tensor de 128 flotantes
            descriptor = recognizer.feature(aligned_face)
            
            print("\n" + "="*40)
            print("📸 DESCRIPTOR CAPTURADO")
            print("Dimensión del array:", descriptor.shape)
            print("Primeros 5 valores:", descriptor[0][:5])
            print("="*40 + "\n")
        else:
            print("\n[!] No se detectó ningún rostro para capturar.\n")

    elif key == ord('q'):
        # Salir si el usuario presiona 'q'
        print("Cerrando aplicación...")
        break

# Liberar los recursos de hardware y destruir ventanas de interfaz de Qt/OpenCV
cap.release()
cv2.destroyAllWindows()