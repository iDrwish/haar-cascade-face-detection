import numpy as np
import cv2


def get_center(x, y, w, h):
    xc = (2 * x + w) / 2
    yc = (2 * y + h) / 2
    return (int(xc), int(yc), int(w / 2))


facePATH = "haarcascade_frontalface_default.xml"
eyePATH = "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(facePATH)
eye_cascade = cv2.CascadeClassifier(eyePATH)

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15,
                                          minNeighbors=5, minSize=(40, 40))
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02,
                                        minNeighbors=20, minSize=(10, 10))

    print(len(faces))
    print(len(eyes))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        xc, yc, rad = get_center(x, y, w, h)
        cv2.circle(img, (xc, yc), rad, (255, 0, 0), 2)

    cv2.imshow("Image", img)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# cv2.imshow("Deteced Faces", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
