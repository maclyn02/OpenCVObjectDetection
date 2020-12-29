import cv2

FACE_DETECTOR = cv2.CascadeClassifier('frontal_face_set.xml')


class FaceDetector:

    def __init__(self):
        pass

    def detect_face_from_image(self):

        # Read an image
        input_image = cv2.imread('v_frontal.jpg')
        # Convert To Grayscale
        grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Run the detector
        faces = FACE_DETECTOR.detectMultiScale(grayscale_image)

        # For each face coordinate draw a rectangle on the input image
        for x, y, w, h in faces:
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # Show the input image(with the rectangle on it)
        cv2.imshow('Image', input_image)

        # Press any key to quit
        cv2.waitKey()

        print('Detection Completed Successfully')

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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

            cv2.imshow('Face', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        webcam.release()
        cv2.destroyAllWindows()

        print('Detection Completed Successfully')
