import cv2

FACE_DETECTOR = cv2.CascadeClassifier('frontal_face_set.xml')
SMILE_DETECTOR = cv2.CascadeClassifier('smile_set.xml')


class SmileDetector:

    def __init__(self):
        pass

    def detect(self):
        webcam = cv2.VideoCapture(0)

        while True:
            # Read a still frame from the video
            successful_frame_read, frame = webcam.read()

            if not successful_frame_read:
                break

            # Convert the frame to gray_scale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face in the gray_frame
            faces = FACE_DETECTOR.detectMultiScale(
                gray_frame, scaleFactor=1.5, minNeighbors=10)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                # crop the face to find the smile
                face = gray_frame[y:y+h, x:x+w]
                # search for smile
                smiles = SMILE_DETECTOR.detectMultiScale(
                    face, scaleFactor=1.5, minNeighbors=30)

                # for x_, y_, w_, h_ in smiles:
                #     cv2.rectangle(frame, (x + x_, y + y_), (x + x_ + w_, y + y_ + h_), (0, 255, 0), 5)

                if len(smiles) > 0:
                    cv2.putText(frame,
                                'Smiling',
                                (x, y+h+50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                5)

            cv2.imshow('Face', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        webcam.release()
        cv2.destroyAllWindows()

        print('Detection Completed Successfully')
