from ultralytics import YOLO
import RPi.GPIO as GPIO
import torch
from time import sleep
from threading import Thread, Event

# import the library
from RpiMotorLib import RpiMotorLib
# Load a model
# model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
# model = YOLO('yolov8n-face.pt')
model = torch.hub.load("ultralytics/yolov5", "yolov5n")
names = model.names

GpioPins_LR = [17, 18, 27, 22]
GpioPins_TB = [19, 16, 13, 12]
fanPin1 = 20
fanPin2 = 26

step_LR = 0
step_TB = 0
STEP = 20
DELAY = 0.001
LIMIT_LR = 70
LIMIT_TB = 70

LOWER_THRESHOLD = 0.42
UPPER_THRESHOLD = 0.58


GPIO.setmode(GPIO.BCM)
GPIO.setup(fanPin1, GPIO.OUT)
GPIO.setup(fanPin2, GPIO.OUT)

GPIO.setup( GpioPins_LR[0], GPIO.OUT )
GPIO.setup( GpioPins_LR[1], GPIO.OUT )
GPIO.setup( GpioPins_LR[2], GPIO.OUT )
GPIO.setup( GpioPins_LR[3], GPIO.OUT )

GPIO.setup( GpioPins_TB[0], GPIO.OUT )
GPIO.setup( GpioPins_TB[1], GPIO.OUT )
GPIO.setup( GpioPins_TB[2], GPIO.OUT )
GPIO.setup( GpioPins_TB[3], GPIO.OUT )

event_LR = Event()
event_TB = Event()

step_sequence = [[1,0,0,1],
                 [1,0,0,0],
                 [1,1,0,0],
                 [0,1,0,0],
                 [0,1,1,0],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,0,0,1]]

def lr(dir):
    global step_LR
    global step_sequence
    motor_LR = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
    for i in range(STEP):
        for pin in range(len(GpioPins_LR)):
            GPIO.output( GpioPins_LR[pin], step_sequence[step_LR % 8][pin] )
        sleep(DELAY)

        if dir:
            step_LR += 1
        else:
            step_LR -= 1

        if event_LR.is_set():
            print('LR kill')
            event_LR.clear()
            return
    

def tb(dir):
    global step_TB
    global step_sequence
    motor_TB = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
    for i in range(STEP):
        for pin in range(len(GpioPins_TB)):
            GPIO.output( GpioPins_TB[pin], step_sequence[step_TB % 8][pin] )
        sleep(DELAY)

        if dir:
            step_TB += 1
        else:
            step_TB -= 1

        if event_TB.is_set():
            print('TB kill')
            event_TB.clear()
            return


def fan(on):
    if on:
        GPIO.output(fanPin1, GPIO.LOW)
        GPIO.output(fanPin2, GPIO.HIGH)
    else:
        GPIO.output(fanPin1, GPIO.LOW)
        GPIO.output(fanPin2, GPIO.LOW)


try:
    # Run batched inference on a list of images
    results = model(0, stream=True, imgsz=240, classes=[0], conf=0.4)  # return a list of Results objects

    # Process results list                
    
    # fan(len(list(results)) > 0)

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            xyxyn = box.xyxyn[0]
            x = (float(xyxyn[0]) + float(xyxyn[2])) / 2
            y = (float(xyxyn[1]) + float(xyxyn[3])) / 2
            
            print(x, y)
            print(step_LR, step_TB)

            event_LR.set()
            event_TB.set()

            t1_available = False
            t2_available = False
            
            if x < LOWER_THRESHOLD and step_LR > -LIMIT_LR:
                while not event_LR.is_set():
                    pass
                t1 = Thread(target=lr, args=(False, ))
                t1.start()
                t1_available = True

            elif x > UPPER_THRESHOLD and step_LR < LIMIT_LR:
                while not event_LR.is_set():
                    pass
                t1 = Thread(target=lr, args=(True, ))
                t1.start()
                t1_available = True

            if y < LOWER_THRESHOLD and step_TB < LIMIT_TB:
                while not event_TB.is_set():
                    pass
                t2 = Thread(target=tb, args=(True, ))
                t2.start()
                t2_available = True

            elif y > UPPER_THRESHOLD and step_TB > -LIMIT_TB:
                while not event_TB.is_set():
                    pass
                t2 = Thread(target=tb, args=(False, ))
                t2.start()
                t2_available = True

            # if t1_available:
            #     t1.join()
            # if t2_available:
            #     t2.join()

            # call the function , pass the parameters

except KeyboardInterrupt:
    GPIO.cleanup()
