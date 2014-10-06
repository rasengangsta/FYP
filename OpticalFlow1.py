#!/usr/bin/env python


from copy import deepcopy

import numpy as np
import cv2
import cv
import time
import sys

help_message = '''
USAGE: OpticalFlow1.py [<video_source>]
Initialising...
'''

def nothing(x):
    pass

def drawProcessedImage(img, lines, outlyingFlowPoints):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    
    for j in range(len(outlyingFlowPoints)):
        for i in range(len(lines[j])):
            point1 = (lines[j][i][0][0], lines[j][i][0][1])
            point2 = (lines[j][i][1][0], lines[j][i][1][1])
            cv2.line(vis, point1, point2, (0, 255, 0), 1) 
            if i in outlyingFlowPoints[j]:
                point1 = (lines[j][i][0][0], lines[j][i][0][1])
                point2 = (point1[0] + 5, point1[1] + 5)
                cv2.rectangle(vis, point1, point2, (255, 0, 0), thickness=8, lineType=8, shift=0)        
    return vis

def differentiateFlow(lines, smoothingConstant):
    pointsX = []
    pointsY = []
    diffPointsX = []
    diffPointsY = []
    counter = 0
    for (x1, y1), (x2, y2) in lines:
        pointsX.append((counter,x2-x1))
        pointsY.append((counter,y2-y1))
        counter = counter + 1
     
    
    pointsSortedX = sorted(pointsX, key=lambda t: t[1])
    pointsSortedY = sorted(pointsY, key=lambda t: t[1])
    
    numPoints = len(pointsX)
 
    for i in range(0, numPoints-smoothingConstant):      
        originalPositionX = pointsSortedX[i][0]
        originalPositionY = pointsSortedY[i][0]
        diffPointsX.append((originalPositionX, pointsSortedX[i+smoothingConstant][1] - pointsSortedX[i][1]))
        diffPointsY.append((originalPositionY, pointsSortedY[i+smoothingConstant][1] - pointsSortedY[i][1]))

    return diffPointsX, diffPointsY

def findOutliers(diffX, diffY, threshold):
    tempX = []
    tempY = []
    for i in range(len(diffX)):
        if diffX[i][1] != 0:
            tempX.append(diffX[i][1])
    
    for i in range(len(diffY)):
        if diffY[i][1] != 0:
            tempY.append(diffY[i][1])
    if len(tempX) == 0:
        tempX.append(0)
    if len(tempY) == 0:
        tempY.append(0)
    minX = min(tempX)
    minY = min(tempY)
    numPoints = len(diffX)
    outlyingPoints = []
    for i in range(numPoints):
        if (minX >= 0 and int(diffX[i][1]) > minX * threshold):   
            outlyingPoints.append(diffX[i][0])
        if (minY >= 0 and int(diffY[i][1]) > minY * threshold):
            outlyingPoints.append(diffY[i][0])
    return outlyingPoints

def quadrantLines(lines):
    topLeft = []
    topRight = []
    botLeft = []
    botRight = []
    for line in lines:     
        if (line[0][0] < 358 and line[0][1] < 235):
            topLeft.append(line)
        elif (line[0][0] > 358 and line[0][1] < 235):
            topRight.append(line)
        elif (line[0][0] < 358 and line[0][1] > 235):
            botLeft.append(line)
        elif (line[0][0] > 358 and line[0][1] > 235):    
            botRight.append(line)
    return (topLeft, topRight, botLeft, botRight)
            
if __name__ == '__main__':   
    print help_message
    try: fn = sys.argv[1]
    except: fn = 0
    cam = cv2.VideoCapture( 'C:\Users\David\Desktop\L2L.avi' )
    fourcc = cv2.cv.CV_FOURCC(*'I420')
    frameDimensions = (int(cam.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('boxe37.avi',fourcc, 10.0, frameDimensions)
    print("Output file opened: " + str(out.isOpened()))
    ret, prev = cam.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')
    cv2.createTrackbar('Threshold','image',1,20,nothing)
    cv2.createTrackbar('Coefficient','image',1,20,nothing)
    cv2.createTrackbar('Vectors','image',6,40,nothing)
    
    while True:

        threshold = cv2.getTrackbarPos('Threshold','image')
        coefficient = cv2.getTrackbarPos('Coefficient','image')
        step = cv2.getTrackbarPos('Vectors','image')
       
        ret, img = cam.read()
        
        if (img is None):
            print("Finnished processing...")
            out.release()
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevGray, gray, 0.5, 3, 15, 3, 2, 1.2, 0)
        prevGray = gray
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.1)
        quads = quadrantLines(lines)
        outlyingFlowPointList = [0]*len(quads)
        
        for i in range(len(quads)):
            diffX, diffY = differentiateFlow(quads[i], coefficient)
            outlyingFlowPointList[i] = (findOutliers(diffX, diffY, threshold))
            
        flowImage = drawProcessedImage(gray, quads, outlyingFlowPointList)
        cv2.imshow('flow', flowImage)
        ch = 0xFF & cv2.waitKey(5)
        

        if ch == 27:
            out.release()
            break
        
    cv2.destroyAllWindows()
