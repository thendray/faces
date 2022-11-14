import cv2


def blur_face(img):
    # получение ширины и высоты
    (h, w) = img.shape[:2]

    # коэффицент размытия
    dW = int(w / 3.0)
    dH = int(h / 3.0)

    if dW % 2 == 0:
        dW -= 1
    if dH % 2 == 0:
        dH -= 1

    return cv2.GaussianBlur(img, (dW, dH), 0)


# Получение доступа к камере
capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

last_face_photo = None
last_coordinates = None

while True:

    ret, img = capture.read()

    # получаем список лиц на фото
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))

    # рисуем прямоугольник на фотографии вокруг лица
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # задание блюра на  фото лица
        img[y:y+h, x:x+w] = blur_face(img[y:y+h, x:x+w])

    if len(faces) > 0:
        last_face_photo = img
        last_coordinates = faces

    cv2.imshow("Video", img)

    # кнопка esc
    button = cv2.waitKey(30) & 0xFF

    if button == 27:
        # на последней фото где есть лицо вырезаем лицо и сохраняем его
        if last_coordinates is not None and last_face_photo is not None:
            for (x, y, w, h) in last_coordinates:
                cv2.imwrite("faces_res/1.jpg", last_face_photo[y:y+h, x:x+w])
        break

capture.release()
cv2.destroyAllWindows()
