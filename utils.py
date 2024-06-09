import cv2
import numpy as np

def obtener_region(frame, region_top, region_bottom, region_left, region_right):
    region = frame[region_top:region_bottom, region_left:region_right]
    return cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (5, 5), 0)

def obtener_promedio(region, fondo, peso_bg):
    if fondo is None:
        fondo = region.copy().astype("float")
    else:
        cv2.accumulateWeighted(region, fondo, peso_bg)
    return fondo

def segmentar(region, fondo):
    diff = cv2.absdiff(fondo.astype(np.uint8), region)
    _, segmented_region = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(segmented_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def escribir_imagen(frame, hand, marcos_transcurridos, tiempo_calibracion, region_left, region_top, region_right, region_bottom):
    if marcos_transcurridos < tiempo_calibracion:
        text = "Calibrando..."
    elif hand and hand.isInFrame:
        if hand.isWaving:
            text = "Saludando"
        if hand.isFist:
            text = "Puño"
        elif hand.isThumbUp:
            text = "Pulgar arriba"
        else:
            text = f"Dedos: {hand.fingers if hand.fingers is not None else 'N/A'}"
    else:
        text = f"Dedos: {hand.fingers if hand and hand.fingers is not None else 'N/A'} - Mano detectada"

    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)
