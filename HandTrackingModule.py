import cv2
import mediapipe as mp
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    device = AudioUtilities.GetSpeakers()
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    volRange = volume.GetVolumeRange()  # range is -96 to 0
    minVol = volRange[0]
    maxVol = volRange[1]
    
    # Adjust these values to fit your hand's range
    minLength = 30   # Minimum distance for volume control
    maxLength = 200  # Maximum distance for volume control
    
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) > 0:
            thumbTip = lmList[4][1:3]
            indexTip = lmList[8][1:3]
            
            if thumbTip and indexTip:
                # Draw circles on tips
                cv2.circle(img, tuple(thumbTip), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, tuple(indexTip), 10, (255, 0, 255), cv2.FILLED)
                
                # Calculate and draw the midpoint
                mx, my = (thumbTip[0] + indexTip[0]) // 2, (thumbTip[1] + indexTip[1]) // 2
                cv2.circle(img, (mx, my), 10, (255, 0, 255), cv2.FILLED)
                
                # Draw line between thumb and index tip
                cv2.line(img, tuple(thumbTip), tuple(indexTip), (255, 0, 255), 3)
                
                # Calculate distance
                length = math.hypot(indexTip[0] - thumbTip[0], indexTip[1] - thumbTip[1])
                
                # Normalize length to percentage (0 to 100)
                lengthPercentage = np.interp(length, [minLength, maxLength], [0, 100])
                
                # Convert percentage to volume level (dB)
                vol = np.interp(lengthPercentage, [0, 100], [minVol, maxVol])
                
                # Set the volume level
                volume.SetMasterVolumeLevel(vol, None)
                
                if length < minLength:
                    cv2.circle(img, (mx, my), 10, (0, 255, 0), cv2.FILLED)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Add a quit condition
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
