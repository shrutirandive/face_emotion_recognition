# Predicting emotions over the static image

from fer import FER
import matplotlib.pyplot as plt
#happy img
img1=plt.imread("1h.jpg")
detector=FER(mtcnn=True)
print(detector.detect_emotions(img1))
plt.imshow(img1)

# If we wish to only want the emotion with the highest score we can directly do that with top_emotion() function.
emotion, score = detector.top_emotion(img1)
print("img1=",emotion,score)

#sad img
img2 = plt.imread("2s.jpg")
detector = FER()
print("img2=",detector.top_emotion(img2))
plt.imshow(img2)

#disgust img
img3 = plt.imread("3d.jpg")
detector = FER()
print("img3=",detector.detect_emotions(img3))
plt.imshow(img3)

#fear img
img4 = plt.imread("4f.jpg")
detector = FER()
print("img4=",detector.detect_emotions(img4))
plt.imshow(img4)

#neutral img
img5 = plt.imread("5n.jpg")
detector = FER()
print("img5=",detector.top_emotion(img5))
plt.imshow(img5)

#surprise img
img6 = plt.imread("6surprise.jpg")
detector = FER()
print("img6=",detector.detect_emotions(img6))
plt.imshow(img6)

#angry img
img7 = plt.imread("7angry.jpg")
detector = FER()
print("img7=",detector.top_emotion(img7))
plt.imshow(img7)