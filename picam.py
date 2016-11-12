import cv2
import numpy as np
import picamera
import io
import time
import os



RESIZE_FACTOR = 4
scaleFactor = 1.1
minNeighbors = 1
minSize = (30,30)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
class OpenCVCapture:

    def __init__(self):
        self.debug_image = "debug.pgm"
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.face_dir = 'face_data'
        if not os.path.isdir(self.face_dir):
            os.mkdir(self.face_dir)
        person_name = raw_input("Enter name:")
        self.face_name = person_name
        self.path = os.path.join(self.face_dir, self.face_name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.model = cv2.createEigenFaceRecognizer()
        self.count_captures = 0

    def read(self):
        
        inKey = raw_input("Input 'c' then enter to register...")
        
        while self.count_captures<10:
           
            if inKey == 'c':
                print "Capturing image...."
                data = io.BytesIO()
                with picamera.PiCamera() as camera:
                    camera.capture(data, format='jpeg')
                data = np.fromstring(data.getvalue(),dtype=np.uint8)
                image = cv2.imdecode(data,1)
                inImg = np.array(image)
                outImg = self.process_image(inImg)
                cv2.imwrite(self.debug_image,outImg)
                time.sleep(0.05)      
        return
    
    def process_image(self,inImg):
        print "processing image"
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (112,92)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray,(gray.shape[1]/RESIZE_FACTOR, gray.shape[0]/RESIZE_FACTOR))
        faces = self.face_cascade.detectMultiScale(
            gray_resized,
            scaleFactor = scaleFactor,
            minNeighbors = minNeighbors,
            minSize = minSize,
            flags = flags
            )
        if len(faces) > 0:
            print "face detected!"
            areas = []
            for(x,y,w,h) in faces:
                areas.append(w*h)
            max_area, idx = max([(val,idx)for idx,val in enumerate(areas)])
            face_sel = faces[idx]

            x = face_sel[0] * RESIZE_FACTOR
            y = face_sel[1] * RESIZE_FACTOR
            w = face_sel[2] * RESIZE_FACTOR
            h = face_sel[3] * RESIZE_FACTOR

            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face,(resized_width,resized_height))
            img_no = sorted([int(fn[:fn.find('.')])for fn in os.listdir(self.path) if fn[0]!='.']+[0])[-1]+1

            cv2.imwrite('%s/%s.pgm' % (self.path,img_no),face_resized)
            print "Count Capture: "+str(self.count_captures)
            self.count_captures +=1
            
        return frame
    
    def eigen_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs,dirs,files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir,subdir)
                for fn in os.listdir(img_path):
                    path = img_path +'/' +fn
                    tag = index
                    imgs.append(cv2.imread(path,0))
                    tags.append(int(tag))
                index +=1
                
            (imgs, tags) = [np.array(item) for item in [imgs,tags]]

            self.model.train(imgs,tags)
            self.model.save('eigen_trained_data.xml')
            print "Successfully trained and save image..."
            return
    

if __name__ =='__main__':
    image_capture = OpenCVCapture()
    image_capture.read()
    image_capture.eigen_train_data()
    
