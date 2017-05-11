import io 
import picamera
import numpy
import cv2
import time

camera = picamera.PiCamera()
camera.hflip = True
camera.vflip = True
camera.resolution = (320, 240)
face_cascade = cv2.CascadeClassifier("/home/pi/Desktop/Projects/haarcascade_frontalface_default.xml")

def takepic(num):
	stream = io.BytesIO()

	camera.capture(stream, format="jpeg")

	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	image = cv2.imdecode(buff, 1)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	print("Found " + str(len(faces)) + " face(s)")
	print(str(faces))

	#for (x, y, w, h) in faces
		#cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

	if len(faces) == 1:
		cropped_image = image[faces[0][1]: faces[0][1] + faces[0][3], faces[0][0]: faces[0][0] + faces[0][2]]
		cv2.imwrite("/will" + str(num).zfill(4) + ".jpeg",  cropped_image)

for i in range(0, 50):
	takepic(i)

