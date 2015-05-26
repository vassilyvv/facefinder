import cv2
import sys
import os
import ImageChops
from PIL import Image
from os import listdir
import hashlib


def find_image_in_db(img):
    for db_img_path in listdir(db_path):
        db_img = cv2.imread(db_path + '/' + db_img_path)
        if not ImageChops.difference(Image.fromarray(db_img), Image.fromarray(img)).getbbox():
            return db_img_path

dirpath = '/'.join(os.path.realpath(__file__).split('/')[:-1])
db_path = dirpath + '/db/'
imagePath = sys.argv[1]
cascadePath = 'haarcascade_frontalface_default.xml'

if not os.path.exists(dirpath + '/db/'):
    os.makedirs(dirpath + '/db/')

faceCascade = cv2.CascadeClassifier(cascadePath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.4,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

cutted_images = [image[face[1]:face[1] + face[3], face[0]: face[0] + face[2]] for face in faces]

for img in cutted_images:
    search_result = find_image_in_db(img)
    if search_result:
        print "Face found: " + search_result
    else:
        m = hashlib.md5()
        m.update(str(img))
        new_path = 'db/%s.png' % m.hexdigest()
        cv2.imwrite(new_path, img)
        print 'New face in database: %s' % new_path