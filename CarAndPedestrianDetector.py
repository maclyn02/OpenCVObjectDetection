import cv2

CARS_DETECTOR = cv2.CascadeClassifier('cars_set.xml')
PEDESTRIAN_DETECTOR = cv2.CascadeClassifier('pedestrians_set.xml')


class CarAndPedestrianDetector:

    def __init__(self):
        pass

    def detect(self):

        webcam = cv2.VideoCapture('road.mp4')

        while True:
            # Read a still frame from the video
            successful_frame_read, frame = webcam.read()

            if not successful_frame_read:
                break

            # Convert the frame to gray_scale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pedestrians = PEDESTRIAN_DETECTOR.detectMultiScale(
                gray_frame, scaleFactor=1.5)

            for x, y, w, h in pedestrians:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

            cars = CARS_DETECTOR.detectMultiScale(
                gray_frame, scaleFactor=1.8)

            for x, y, w, h in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

            cv2.imshow('Image', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        webcam.release()
        cv2.destroyAllWindows()
        print('Detection Completed Successfully')
