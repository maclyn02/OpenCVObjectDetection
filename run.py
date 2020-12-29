import inquirer
from FaceDetector import FaceDetector
from CarAndPedestrianDetector import CarAndPedestrianDetector
from SmileDetector import SmileDetector
from EyesDetector import EyesDetector

module_selection = [
    inquirer.List('ModuleName',
                  message="Which python script do you want to run?",
                  choices=['FaceDetector',
                           'CarAndPedestrianDetector', 'SmileDetector', 'EyesDetector'],
                  ),
]
answer = inquirer.prompt(module_selection)

if answer['ModuleName'] == 'FaceDetector':
    detector = FaceDetector()
elif answer['ModuleName'] == 'CarAndPedestrianDetector':
    detector = CarAndPedestrianDetector()
elif answer['ModuleName'] == 'SmileDetector':
    detector = SmileDetector()
elif answer['ModuleName'] == 'EyesDetector':
    detector = EyesDetector()

detector.detect()
