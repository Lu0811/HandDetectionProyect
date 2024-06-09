import numpy as np
import cv2

class DatosMano:
    def __init__(self, top, bottom, left, right, centerX):
        self.center_positions = []
        self.actualizar(top, bottom, left, right, centerX)
        self.prevCenterX = 0
        self.isInFrame = False
        self.fingers = None
        #* Gestures
        self.isWaving = False
        self.isThumbUp = False
        self.isFist = False
        self.gestureList = []

    def actualizar(self, top, bottom, left, right, centerX):
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        self.centerX = centerX
        self.center_positions.append(centerX)
        if len(self.center_positions) > 10:
            self.center_positions.pop(0)
        self.isInFrame = True

    def verificar_saluda(self):
        if len(self.center_positions) >= 2:
            movement = np.abs(self.center_positions[-1] - self.center_positions[0])
            self.isWaving = movement > 20  # Ajusta este umbral según sea necesario

    def verificar_pulgar_arriba(self):
        if self.fingers == 1:
            if self.top[1] < self.centerY - 20:  
                self.isThumbUp = True

    def verificar_fist(self):
        if self.fingers == 0:
            if self.right[0] - self.left[0] < 50:  
                self.isFist = True

def obtener_datos_mano(segmented_image, frame, roi_offset, mano):
    convexHull = cv2.convexHull(segmented_image)
    top = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    centerX = int((left[0] + right[0]) / 2)
    centerY = int((top[1] + bottom[1]) / 2)
    center = (centerX, centerY)

    if mano is None:
        mano = DatosMano(top, bottom, left, right, centerX)
    else:
        mano.actualizar(top, bottom, left, right, centerX)
    mano.verificar_saluda()
    mano.verificar_pulgar_arriba()
    mano.verificar_fist()

    fingers, finger_tips, finger_bases = contar_dedos(segmented_image, frame, roi_offset, center)
    mano.gestureList.append(fingers)
    if len(mano.gestureList) % 12 == 0:
        mano.fingers = most_frequent(mano.gestureList)
        mano.gestureList.clear()

    dibujar_esqueleto(frame, finger_tips, finger_bases, center, roi_offset)
    return mano

def contar_dedos(segmented_image, frame, roi_offset, center):
    epsilon = 0.02 * cv2.arcLength(segmented_image, True)
    approx = cv2.approxPolyDP(segmented_image, epsilon, True)

    if len(approx) < 3:
        return 0, [], []

    try:
        chull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, chull)
    except cv2.error as e:
        print(f"Error en convexityDefects: {e}")
        return 0, [], []

    if defects is None:
        return 0, [], []

    finger_tips = []
    finger_bases = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start, end, far = tuple(approx[s][0]), tuple(approx[e][0]), tuple(approx[f][0])
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(far) - np.array(end))
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        if angle <= np.pi / 2 and d > 30:
            finger_tips.append(start)
            finger_bases.append(far)
            finger_tips.append(end)

    finger_tips = list(set(finger_tips))
    finger_tips.sort(key=lambda x: x[0])
    finger_bases = list(set(finger_bases))
    finger_bases.sort(key=lambda x: x[0])

    return len(finger_tips), finger_tips, finger_bases

def dibujar_esqueleto(frame, finger_tips, finger_bases, center, roi_offset):
    center = (center[0] + roi_offset[0], center[1] + roi_offset[1])
    for base in finger_bases:
        base = (base[0] + roi_offset[0], base[1] + roi_offset[1])
        cv2.line(frame, center, base, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        cv2.circle(frame, base, 5, (0, 0, 255), -1)

    for tip in finger_tips:
        tip = (tip[0] + roi_offset[0], tip[1] + roi_offset[1])
        cv2.circle(frame, tip, 5, (0, 255, 255), -1)
        for base in finger_bases:
            base = (base[0] + roi_offset[0], base[1] + roi_offset[1])
            if np.linalg.norm(np.array(tip) - np.array(base)) < 100:
                cv2.line(frame, base, tip, (0, 255, 0), 2)

def most_frequent(input_list):
    return max(set(input_list), key=input_list.count)
