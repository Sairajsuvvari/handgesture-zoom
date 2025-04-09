import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

cap.set(3, 780)
cap.set(4, 1280)
detector = HandDetector(detectionCon=0.8)

startDist = None
scale = 0
cx, cy = 500, 500
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread('meta-6871457_640.jpg')

    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

            if startDist is None:
                startDist = length

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]

            # Display the distance between the hands
            cv2.putText(img, f"Distance: {length}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            startDist = None
    else:
        startDist = None

    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))

        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)