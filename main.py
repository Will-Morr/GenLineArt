import math as m
import numpy as np
import scipy as sp
from scipy.ndimage import convolve
import os
import sys
import time
from copy import deepcopy
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt 


from funcs import *
from defs import *

if __name__ == '__main__':
    folderName = sys.argv[1]
	
    filePath = 'proc/' + folderName
    
    # Check if input exists
    if not os.path.exists(filePath + '/input.jpg'):
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        # Take picture

    # Load image
    inImg = Image.open(filePath + '/input.jpg').convert('RGB')

    # Resize image
    origImSize = inImg.size
    if MAX_DIMS[0] / origImSize[0] > MAX_DIMS[1] / origImSize[1]:
        MAX_DIMS[1] = int(origImSize[1] * MAX_DIMS[0] / origImSize[0])
        # inImg.resize((MAX_DIMS[0], ), Image.Resampling.HAMMING)
    else:
        MAX_DIMS[0] = int(origImSize[0] * MAX_DIMS[1] / origImSize[1])
    inImg = inImg.resize(MAX_DIMS, Image.Resampling.HAMMING)

    # Load array of raw pixels
    rawImg = np.array(inImg.convert('L'))
    saveArbitraryImage(rawImg, filePath+'/img_rawGrayScale.jpg')

    # Plot hist of input image values if requested
    plt.hist(rawImg.flatten(), bins=256)
    # plt.axvline(MAX_VAL, color='orange')
    plt.title("Distribution of shade in raw image")
    plt.savefig(filePath + '/plt_pixDistribution.jpg')

    # Linearize image
    if DO_IMAGE_LINEARIZATION:
        rawImg = linearizeImage(rawImg)

    # Identify white background
    pixelWhiteness = np.array(rawImg, dtype=np.double)
    colPix = np.array(inImg)
    if STD_IGNORE_BLUE:
        stdev = np.std(colPix[:, :, :2], axis=2)
    else:
        stdev = np.std(colPix, axis=2)
    
    
    # Plot image STDev distribution
    plt.cla()
    plt.hist(np.array(stdev).flatten(), bins=50)
    plt.title("Standard deviation of RGB of each pixel")    
    plt.savefig(filePath + '/plt_pixStd.jpg')

    # Set all stdevs < 5 to 5
    stdev[stdev < STD_TRUNC_VAL] = STD_TRUNC_VAL
    stdev /= STD_TRUNC_VAL

    saveArbitraryImage(stdev, filePath+'/img_stdev.jpg')

    # Actually adjust for whiteness
    pixelWhiteness /= stdev
    backgroundVal = np.min(pixelWhiteness[:2, :])
    rawImg[pixelWhiteness > backgroundVal] = 255

    # Plot pixel whiteness
    pixelWhiteness[pixelWhiteness > backgroundVal] *= 4
    saveArbitraryImage(pixelWhiteness, filePath+'/img_whiteness.jpg')

    # Approximate gradient

    def convolve2dToNd(arr, kernel):
        outArr = np.zeros_like(arr)
        for ii in range(arr.shape[2]):
            outArr[:, :, ii] = convolve(arr[:, :, ii], kernel)
        return outArr
    
    sobelKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SOBEL_KERNEL_RAD*2+1, SOBEL_KERNEL_RAD*2+1))
    sobelKernel = np.array(sobelKernel, dtype=np.int16)
    sobelKernel[:, SOBEL_KERNEL_RAD] = 0
    sobelKernel[:, :SOBEL_KERNEL_RAD] *= -1

    sobelSmoothKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SOBEL_PRE_SMOOTH_RAD*2+1, SOBEL_PRE_SMOOTH_RAD*2+1))
    
    if SOBEL_DO_RGB:
        rawImg_smoothed = convolve2dToNd(np.array(inImg, dtype=np.int32), sobelSmoothKernel) / np.sum(sobelSmoothKernel)

        sobelHorz = convolve2dToNd(np.array(rawImg_smoothed, dtype=np.int32), sobelKernel)
        # sobelHorz = np.sum(np.abs(sobelHorz), axis=2)
        sobelHorz = np.sum(sobelHorz, axis=2)
        sobelKernel = np.swapaxes(sobelKernel, 0, 1)
        sobelVert = convolve2dToNd(np.array(rawImg_smoothed, dtype=np.int32), sobelKernel)
        sobelVert = np.sum(np.abs(sobelVert), axis=2)
    else:
        rawImg_smoothed = convolve(np.array(rawImg, dtype=np.int32), sobelSmoothKernel) / np.sum(sobelSmoothKernel)

        sobelHorz = convolve(np.array(rawImg_smoothed, dtype=np.int32), sobelKernel)
        # sobelHorz = np.abs(sobelHorz)
        sobelKernel = np.swapaxes(sobelKernel, 0, 1)
        sobelVert = convolve(np.array(rawImg_smoothed, dtype=np.int32), sobelKernel)
        sobelVert = np.abs(sobelVert)


    sobelImg = np.zeros_like(inImg, dtype=np.int32)
    sobelImg[:, :, 0] = sobelHorz + np.min(sobelHorz)
    sobelImg[:, :, 1] = sobelVert + np.min(sobelVert)
    saveArbitraryImage(sobelImg, filePath+'/img_sobel.jpg', mode='RGB')


    sobelMag = np.abs(sobelHorz) + np.abs(sobelVert)
    sobelDir = np.arctan2(sobelHorz, sobelVert)

    sobelNetSmoothingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SOBEL_NET_SMOOTH_RAD*2+1, SOBEL_NET_SMOOTH_RAD*2+1))
    sobelMag = convolve(sobelMag, sobelNetSmoothingKernel)


    # sobelHorz = np.array(sobelHorz, dtype=np.double) / (sobelMag + np.average(sobelMag))
    # sobelVert = np.array(sobelVert, dtype=np.double) / (sobelMag + np.average(sobelMag))

    # sobelImg = np.zeros_like(inImg, dtype=np.double)
    # sobelImg[:, :, 0] = sobelHorz
    # sobelImg[:, :, 1] = sobelVert
    saveArbitraryImage(sobelDir - np.min(sobelDir), filePath+'/img_sobelAdjusted.jpg')
    # saveArbitraryImage(sobelDir + np.min(sobelDir), filePath+'/img_sobelAdjusted.jpg', mode='RGB')
    
    adjustImg = np.array(rawImg, dtype=np.double)
    adjustImg /= np.max(adjustImg)

    if DARKEN_NOT_WHITE:
        adjustImg += (stdev + STD_TRUNC_VAL)/300
        
    if DARKEN_EDGES:
        print("Handling edges")
        smoothKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SMOOTH_KERNEL_RAD*2+1,SMOOTH_KERNEL_RAD*2+1))
        diffKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(EDGE_KERNEL_RAD*2+1,EDGE_KERNEL_RAD*2+1))
        modColPix = np.array(colPix, dtype=np.int32)
        colDiffs = np.zeros_like(modColPix, dtype=np.double)
        
        for ii in range(3):
            modColPix[:, :, ii] = convolve(modColPix[:, :, ii], smoothKernel) / np.sum(smoothKernel)
            colDiffs[:, :, ii] = convolve(modColPix[:, :, ii], diffKernel)#, mode='same')
        colDiffs = modColPix - colDiffs/np.sum(diffKernel)
        # colDiffs -= np.min(colDiffs)

        # Plot image edge selection distribution
        plt.cla()
        plt.hist(np.sum(colDiffs, axis=2).flatten(), bins=50)
        plt.title("Absolute sum of RGB Difference in each axis")
        plt.savefig(filePath + '/plt_edgeDist.jpg')

        colDiffs[colDiffs > 0] = 0

        colDiffs = np.max(np.abs(colDiffs), axis=2)
        colDiffs[colDiffs > 100] = 100
        saveArbitraryImage(colDiffs, filePath+'/img_edges.jpg', mode='L')
        
        # adjustImg -= colDiffs/500
        # adjustImg = np.min([adjustImg, 1.0-colDiffs/150], axis=0)
        adjustImg -= EDGE_DARKENING_RATIO*colDiffs/100
        adjustImg[adjustImg < -0.0] = -0.0

    # Save greyscale image
    # saveArbitraryImage(adjustImg, filePath+'/img_tmp.jpg', mode='L')
    whitePts = np.where(rawImg == 255)
    adjustImg -= np.min(adjustImg)
    adjustImg *= 255/np.max(adjustImg)
    rawImg = np.array(adjustImg, dtype=np.uint8)
    rawImg[whitePts] = 255
    Image.fromarray(rawImg, mode='L').save(filePath + '/img_grey.jpg')

    # Plot image edge selection distribution
    plt.cla()
    shadePts = rawImg.flatten()
    plt.hist(shadePts[shadePts < 255], bins=256)
    plt.title("Distribution of shade in final greyscale image")
    plt.savefig(filePath + '/plt_shadeDist.jpg')


    # exit()

    # Convert to pts
    pointMap = convertToPoints(rawImg, skipToEveryNth=SKIP_TO_NTH, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD, maxVal=MAX_VAL, minDist=MIN_DIST, scaleFact=SCALE_FACT)

    # Connect lines

    # Display point output
    points = Image.new("RGB", inImg.size, "white")
    draw = ImageDraw.Draw(points)
    for pt in iterateOverFullMap(pointMap):
        draw.ellipse([(pt[1]-CIRCLE_RAD, pt[0]-CIRCLE_RAD), (pt[1]+CIRCLE_RAD, pt[0]+CIRCLE_RAD)], fill="black", )
    points.save(filePath + '/out_points.jpg')
    # points.show()
    

    # Export points
    doc = ezdxf.new()
    msp = doc.modelspace()
    for pt in iterateOverFullMap(pointMap):
        msp.add_circle((pt[1]*MM_PER_PIX, -pt[0]*MM_PER_PIX), CIRCLE_RAD)

    doc.saveas(filePath+'/out_points.dxf')
    doc.saveas('proc/currCircleArt.dxf')

    # Display tangent lines
    sobelDir += np.pi/2
    xMagSet = np.cos(sobelDir)
    yMagSet = np.sin(sobelDir)
    
    tanLines = []
    for pt in iterateOverFullMap(pointMap):
        xMag = xMagSet[*pt]
        yMag = yMagSet[*pt]
        tanLines.append([(pt[0] - xMag*TAN_LINE_RAD, pt[1] - yMag*TAN_LINE_RAD), (pt[0] + xMag*TAN_LINE_RAD, pt[1] + yMag*TAN_LINE_RAD)])
    exportLines(tanLines, filePath+'/out_tanLines', inImg, MM_PER_PIX)

    # Generate lines
    lineData = connectPoints(pointMap, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD, maxLineLen=MAX_LINE_LEN)
    # lineData = pointsToLines(pointMap, subDivRad=SUBDIV_RAD, maxLineLen=MAX_LINE_LEN)

    # Plot lines
    outLines = []
    for fooLine in lineData:
        for idx in range(len(fooLine)-1):
            outLines.append([(fooLine[idx][0], fooLine[idx][1]), (fooLine[idx+1][0], fooLine[idx+1][1])])
    exportLines(outLines, filePath+'/out_linesout_lines', inImg, MM_PER_PIX)
    exportLines(outLines, 'proc/currLineArt', inImg, MM_PER_PIX)

    # Generate lines
    lineData = connectPointsWithTangents(pointMap, sobelDir, sobelMag, sobelFactor=SOBEL_MULT, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD, maxLineLen=MAX_LINE_LEN)
    # lineData = pointsToLines(pointMap, subDivRad=SUBDIV_RAD, maxLineLen=MAX_LINE_LEN)

    # Plot lines
    pointLines = []
    for fooLine in lineData:
        for idx in range(len(fooLine)-1):
            pointLines.append([(fooLine[idx][0], fooLine[idx][1]), (fooLine[idx+1][0], fooLine[idx+1][1])])

    exportLines(pointLines, filePath+'/out_tanLinesV2', inImg, MM_PER_PIX)
    exportLines(pointLines, 'proc/currLineArt', inImg, MM_PER_PIX)



    # Generate lines with tangent handling
    # lineData = connectPoints(pointMap, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD, maxLineLen=MAX_LINE_LEN)


