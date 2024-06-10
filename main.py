import cv2
import numpy as np
from hand_detection import DatosMano, obtener_datos_mano
from utils import obtener_region, obtener_promedio, segmentar, escribir_imagen

ALTURA_FRAME = 480
ANCHO_FRAME = 640
TIEMPO_CALIBRACION = 30

fondo = None
marcos_transcurridos = 0

region_top, region_bottom, region_left, region_right = 0, int(2 * ALTURA_FRAME / 3), int(ANCHO_FRAME / 2), ANCHO_FRAME
mano = None

camera_index = 0
capture = cv2.VideoCapture(camera_index)
if not capture.isOpened():
    capture = cv2.VideoCapture(1)
if not capture.isOpened():
    raise Exception("No se pudo abrir el dispositivo de video")

cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Input", 1280, 720)

while True:
    ret, frame = capture.read()
    if not ret:
        print("No se pudo capturar el cuadro")
        break
    frame = cv2.resize(frame, (ANCHO_FRAME, ALTURA_FRAME))
    frame = cv2.flip(frame, 1)
    region = obtener_region(frame, region_top, region_bottom, region_left, region_right)

    if marcos_transcurridos < TIEMPO_CALIBRACION:
        fondo = obtener_promedio(region, fondo, 0.5)
    else:
        segmented_region = segmentar(region, fondo)
        if segmented_region is not None:
            cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
            cv2.imshow("Segmented Image", region)
            mano = obtener_datos_mano(segmented_region, frame, np.array([region_left, region_top]), mano)

            if mano.isPeaceSign:
                cv2.putText(frame, "Simbolo de Paz", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            if mano.isStopSign:
                cv2.putText(frame, "Simbolo de Stop", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if mano.isLike:
                cv2.putText(frame, "Like", (10, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            if mano.isOk:
                cv2.putText(frame, "OK", (10, 180), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if mano.isMiddleFinger:
                cv2.putText(frame, "Dedo Corazon", (10, 220), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    escribir_imagen(frame, mano, marcos_transcurridos, TIEMPO_CALIBRACION, region_left, region_top, region_right, region_bottom)
    cv2.imshow("Camera Input", frame)
    marcos_transcurridos += 1
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

capture.release()
cv2.destroyAllWindows()
