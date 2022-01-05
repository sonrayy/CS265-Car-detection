#ref : car detection 
#-> https://github.com/amartya-k/vision

#ref : object detection
#-> https://www.youtube.com/watch?v=GGeF_3QOHGE&list=RDCMUCYUjYU5FveRAscQ8V21w81A&index=1
#-> https://www.youtube.com/watch?v=9AycYn9gj1U&list=RDCMUCYUjYU5FveRAscQ8V21w81A&index=2
#-> https://www.youtube.com/watch?v=xK4li3jinSw&list=RDCMUCYUjYU5FveRAscQ8V21w81A&index=3
#-> https://www.youtube.com/watch?v=Sp9mEGubBJs&list=RDCMUCYUjYU5FveRAscQ8V21w81A&index=4

#ref : yolo
#-> https://pjreddie.com/darknet/yolo/ (for yolov3)
#-> https://github.com/kiyoshiiriemon/yolov4_darknet (for yolov4)

#ref : coco.names
#-> https://github.com/pjreddie/darknet/blob/master/data/coco.names

# infomation for some command 
# -> https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html 
# -> https://docs.opencv.org/3.4/d6/d0f/group__dnn.html

# Adapted : SoalakAI (AI in Government)
# CS265 project

import cv2
import numpy as np
import vehicles # from vehicles.py
import time

cnt_up=0
cnt_down=0
cnt_all = 0

whT = 320 # w = h = 320 -> declared th width and the height of the target to 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []

# read class names from coco.names and split with '\n'
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# model that we want to use.
# tiny -> faster but less accurate
# else -> slowe but more accurate
'''
modelConfiguration = 'yolo/yolov4-tiny.cfg'
modelWeights = 'yolo/yolov4-tiny.weights'
'''

modelConfiguration = 'yolo/yolov3-tiny.cfg'
modelWeights = 'yolo/yolov3-tiny.weights'


# Reads a network model that stored in Darknet model files.
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
# use OpenCV for backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# use CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    # bounding box -> contain x,y,w,h
    bbox = []
    # contain all class ids
    classIds = []
    # confidence values
    confs = []

    for output in outputs: # we got 3 ouputs -> go through each one
        for detect in output:
            scores = detect[5:] #remove the fist five elements
            classId = np.argmax(scores) # find the max confidence index
            confidence = scores[classId] # find that confidence value
            if confidence > confThreshold: # if the confidence value > 0.5 -> good detection
                #save the x,y,w,h
                w,h = int(detect[2]*wT), int(detect[3]*hT) # multiply with wT -> from percentage to actual pixel, height is the same process
                x,y = int(detect[0]*wT - w/2), int(detect[1]*hT - h/2) # divided the width by 2 and subtract from x*wT -> for the actual center x pixel, center y is the same process
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # eluminated the overlapping boxes; nmsTreshold 
    # -> high more overlapping boxes    
    # -> low less overlapping boxes    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        # draw a box 
        #cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,0),thickness=2)
        #cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,255),thickness=1)
        #change color for differnt obj
        if (classNames[classIds[i]] == 'person'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1) # white
        elif (classNames[classIds[i]] == 'bicycle'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1) # red
        elif (classNames[classIds[i]] == 'car'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1) # green
        elif (classNames[classIds[i]] == 'motorbike'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1) # blue
        elif (classNames[classIds[i]] == 'bus'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1) # purple
        elif (classNames[classIds[i]] == 'train'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),1) # orange
        elif (classNames[classIds[i]] == 'truck'):
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(119,136,153),1) # light slate gray
        else:
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
         cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1) # light slate gray
        
        
def classify(frame):
    # Convert the image to blob because the network model accept a particular type of format which is blob
    blob = cv2.dnn.blobFromImage(frame,1/255,(whT,whT),[0,0,0],1,crop=False)
    # sent the blob to our network model
    net.setInput(blob)

    # our network model architecture is do the convolution first(many times) and pass the layer of predicted output (in this case is 3) 
    # which is differcnt in each one -> so we want to know those layers of output names
    layerNames = net.getLayerNames()

    # extract only the output layers
    # we are not get the names yet, we get the index -> use this to reference to the layer names
    # and get the output names
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    outputs = net.forward(outputNames)
    # each output is a nupy array -> print out the shape
    # outputs[0] -> (300, 85)
    # outputs[1] -> (1200, 85)
    # outputs[2] -> (4800, 85)
    # the x coordinate (300,...) is the number of bounding boxes
    # the y coordinate (85) is all the class names (80 classes) with
    # canter x, center y, width ,height, confidence (5 values added)
    # confidence is the propability of the predicted obj that correct -> e.g. confidence is .93 then that predicted object is 93% chance to be correct
    findObjects(outputs,frame)

# videos for testing

#cap=cv2.VideoCapture("test_video/Freewa.mp4")
#cap=cv2.VideoCapture("test_video/surveillance.m4v")
#cap=cv2.VideoCapture("test_video/diy_test.mp4")
cap=cv2.VideoCapture("test_video/video.mp4")
#cap=cv2.VideoCapture("test_video/Highway Free footage.mp4")
#cap=cv2.VideoCapture("test_video/footage3.mp4")

#cap=cv2.VideoCapture("test_video/sonrayy_footage_1 (convert-video-online.com).mp4")
#cap=cv2.VideoCapture("test_video/sonrayy_footage_2 (convert-video-online.com).mp4")
#cap=cv2.VideoCapture("test_video/sonrayy_footage_3 (convert-video-online.com).mp4")


#cap = cv2.VideoCapture(0)

'''
# for image classification

img = cv2.imread('usain.jpg')
classify(img)
cv2.imshow('Usain Bolt', img)
cv2.waitKey(0)
'''

#Get width and height of video
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/3))
line_down=int(2*(h/3))

up_limit=int(line_up - 150)
down_limit=int(line_up + 150)

print("Red line y:",str(line_down))
print("Blue line y:",str(line_up))

line_down_color=(0,0,255) # set line colour to red (b,g,r)
line_up_color=(255,0,0) # set line colour to blue (b,g,r)

pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8) # use for opening
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8) # use for closing


font = cv2.FONT_HERSHEY_DUPLEX 
cars = []
max_p_age = 5
pid = 1


while(cap.isOpened()):
    ret,frame=cap.read()

    classify(frame) # classification

    for i in cars:
        i.age_one()
        
    fgmask=fgbg.apply(frame)
    fgmask2=fgbg.apply(frame)

    if ret==True:

        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)


        #Find Contours
        countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
            #print(area)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h: # if car is in the video
                            new = False
                            i.updateCoords(cx, cy) # then update the coordinate

                            if i.going_UP(line_down,line_up)==True: # if it go up
                                cnt_up+=1 
                                cnt_all+=1
                                cv2.polylines(frame, [pts_L2], False, line_down_color, thickness=3) # change the colour of the checking line -> make it flashing
                                #print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                            elif i.going_DOWN(line_down,line_up)==True: # same as go up
                                cnt_down+=1
                                cnt_all+=1
                                cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=3)
                                #print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            break
                        if i.getState()=='1': # if the car was passed the checking line
                            if i.getDir()=='down'and i.getY()>down_limit: # and the car was passed the down limit line
                                i.setDone() # then set done -> do not count anymore
                            elif i.getDir()=='up'and i.getY()<up_limit: # same as above
                                i.setDone()
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i

                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1) # draw a red circle in the centre
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # draw a green regtangle around the detected car

        #for i in cars:
        #    cv2.putText(frame, str(i.getRGB()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_up='UP: '+str(cnt_up) #create a string to show how many cars going up
        str_down='DOWN: '+str(cnt_down) #create a string to show how many cars going up
        str_all = 'ALL:'+str(cnt_all) #create a string to show how many were detected

        frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2) # draw the going-up checking line
        frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2) # draw the going-down checking line
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1) # draw going-up limit line
        frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1) # draw going-down limit line

        cv2.putText(frame, str_up, (10, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_up, (10, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (160, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_down, (160, 40), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str_all, (360, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_all, (360, 40), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Press "Q" to exit.', (10, 80), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, 'Press "Q" to exit.', (10, 80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xff==ord('q'):
            break

    else:
        break
# print conclusion in terminal when exit.
print()
print('CONCLUSION')
print(' UP:',int(cnt_up))
print(' DOWN:',int(cnt_down))
print(' ALL:',int(cnt_all))

cap.release()
cv2.destroyAllWindows()

#create more story
