import cv2
import sys
import ImageChops
from PIL import Image
from os import listdir

db_path = '/home/zuber/projects/facepalm/db'
imagePath = sys.argv[1]
cascPath = sys.argv[2]

def find_image_in_db(img):
    for db_img_path in listdir(db_path):
        db_img = cv2.imread(db_path + '/' + db_img_path)
        if not ImageChops.difference(Image.fromarray(db_img), Image.fromarray(img)).getbbox():
            return db_img_path

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.4,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Draw a rectangle around the faces
#for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
cutted_images = [image[face[1]:face[1] + face[3], face[0]: face[0] + face[2]] for face in faces]
for img in cutted_images:
    search_result = find_image_in_db(img)
    if search_result:
        print "face found: " + search_result
    else:
        print 'j'
        cv2.imwrite('db/img%d.png' % (len(listdir(db_path)) + 1), img)