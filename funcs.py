import math as m
import numpy as np
import os
import sys
import time
from copy import deepcopy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 
import ezdxf


# Shorthand for magnitude of vector
def magnitude(vect):
    return(np.sqrt(np.sum(np.power(vect,2))))
	# if len(vect.shape) == 2:
	# 	return(np.sqrt(np.sum(pow(vect,2), axis=0)))
	# else:
	# 	return(np.sqrt(np.sum(pow(vect,2))))

def saveArbitraryImage(arr, path, mode='L'):
    washOutImage = Image.fromarray(np.array(((arr-np.min(arr))/(np.max(arr)-np.min(arr)))*255, dtype=np.uint8), mode = mode)
    washOutImage.save(path)

def linearizeImage(arr):
    vals, count = np.unique(arr, return_counts=True)
    
    totalCount = np.sum(count)
    valFractions = np.zeros((256), np.double)
    valFractions[vals] = count/totalCount
    
    sumValFracs = np.zeros((256), np.uint8)
    for jj in range(256):
        sumValFracs[jj] = np.sum(valFractions[0:jj+1])*256

    # sumValFracs[1:] = (sumValFracs[:-1] + sumValFracs[1:]) / 2

    return sumValFracs[arr]

# Make empty subdivision map
def initRawSubdivMap(shape, subdivSize):
    return [[[] for jj in range(int(shape[1]/subdivSize + 1.0))] for ii in range(int(shape[0]/subdivSize + 1.0))]

# Iterate over every object in range on map
def subDivMapIterator(map, pos, rad, includeIdx=False):
    xInd = pos[0]
    yInd = pos[1]
    xSize = len(map)
    ySize = len(map[1])
    for xx in range(xInd-rad, xInd+rad+1):
        if xx < 0: continue
        if xx >= xSize: continue
        
        for yy in range(yInd-rad, yInd+rad+1):
            if yy < 0: continue
            if yy >= ySize: continue


            if includeIdx:
                idx = 0
                while idx < len(map[xx][yy]):
                    yield map[xx][yy][idx], (xx, yy, idx)
                    idx += 1
            else:
                for foo in map[xx][yy]:
                    yield foo

def iterateOverFullMap(map, destroy=False):
    for xSet in map:
        for ySet in xSet:
            for fooObj in ySet:
                yield fooObj
                if destroy:
                    ySet.remove(fooObj)
    
# Convert RGB image to points
def convertToPoints(imgArr, skipToEveryNth = 10, subdivSize = 50, subDivRad = 1, maxVal=200, minDist=10, scaleFact=4.0):
    # Load input data
    iW, iH = imgArr.shape

    pointMap = initRawSubdivMap(imgArr.shape, subdivSize)
    
    pixTestList = np.meshgrid(np.arange(0, iW, skipToEveryNth), np.arange(0, iH, skipToEveryNth))
    pixTestList = np.array([foo.flatten() for foo in pixTestList])
    valList = imgArr[*pixTestList]

    sortList = np.argsort(valList)
    valList = valList[sortList]
    pixTestList = pixTestList[:, sortList]

    for xx, yy in np.swapaxes(pixTestList, 0, 1):
        val = imgArr[xx, yy]

        if val > maxVal:
            continue

        doPlace = True
        xMapInd = int(xx/subdivSize)
        yMapInd = int(yy/subdivSize)
        for fooPt in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad):
            dist = magnitude([fooPt[0] - xx, fooPt[1] - yy])

            if dist < scaleFact * float(val)/maxVal + minDist:
                doPlace = False
                break

        if doPlace:
            pointMap[xMapInd][yMapInd].append([xx, yy])
            print(val, ' ', xx, ' ', yy)
    return(pointMap)

def pointsToLines(inputPointMap, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for xx in range(len(pointMap[0])):
        for yy in range(len(pointMap[1])):
            idx = 0
            while idx < len(pointMap[xx][yy]):
                # Starting new line
                currIdx = (xx, yy, idx)
                currPt = pointMap[xx][yy][idx]
                outLines.append([])
                print(len(outLines))


                # Iterate until we fail to find a match
                while True:
                    minDist = maxLineLen
                    bestPt = None
                    bestIdx = None

                    # print(currIdx)
                    # print(len(pointMap[currIdx[0]][currIdx[1]]))

                    pointMap[currIdx[0]][currIdx[1]].pop(currIdx[2])
                    outLines[-1].append(currPt)

                    for cmpPt, cmpIdx in subDivMapIterator(pointMap, (currIdx[0], currIdx[1]), subDivRad, includeIdx=True):
                        dist = magnitude([currPt[0] - cmpPt[0], currPt[1] - cmpPt[1]])
                        if dist < minDist:
                            minDist = dist
                            bestPt = cmpPt
                            bestIdx = cmpIdx
                    

                    if bestPt != None:
                        print(minDist)
                        currIdx = bestIdx
                        currPt = bestPt
                    else:
                        if len(outLines[-1]) <= 1:
                            outLines.pop(-1)
                        break
    return outLines

def connectPoints(inputPointMap, subdivSize=50, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for currPt in iterateOverFullMap(pointMap, destroy=True):
        xMapInd = int(currPt[0]/subdivSize)
        yMapInd = int(currPt[1]/subdivSize)
    
        minDist = maxLineLen
        bestPt = None

        for cmpPt, cmpIdx in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad, includeIdx=True):
            if currPt == cmpPt: 
                continue
            dist = magnitude([currPt[0] - cmpPt[0], currPt[1] - cmpPt[1]])
            if dist < minDist:
                print(dist)
                minDist = dist
                bestPt = cmpPt

        if bestPt != None:
            # if [currPt, bestPt] not in outLines:
            outLines.append([currPt, bestPt])

    return outLines



def connectPointsWithTangents(inputPointMap, sobelDir, sobelMag, sobelFactor = 10.0, subdivSize=50, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for currPt in iterateOverFullMap(pointMap, destroy=True):
        currPtNp = np.array(currPt)
        sobelDirPt = sobelDir[*currPtNp]
        sobelMagPt = sobelMag[*currPtNp]

        xMapInd = int(currPt[0]/subdivSize)
        yMapInd = int(currPt[1]/subdivSize)
    
        minDist = np.inf
        bestPt = None

        print('\n---------')
        for cmpPt, cmpIdx in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad, includeIdx=True):
            if currPt == cmpPt: 
                continue
            diff = [cmpPt[0] - currPt[0], cmpPt[1] - currPt[1]]
            ang = np.arctan2(*diff)
            
            angDiff =  sobelDirPt - ang
            while angDiff > np.pi: angDiff -= np.pi*2
            angDiff = np.abs(angDiff)

            print('   ', sobelDirPt, ' ', diff, ' -> ', ang, '   ', angDiff)

            dist = magnitude(diff)

            score = dist + angDiff*sobelFactor
            if score < minDist:
                # print(dist)
                minDist = dist
                bestPt = cmpPt

        if bestPt != None:
            # if [currPt, bestPt] not in outLines:
            outLines.append([currPt, bestPt])
            
    return outLines

def exportLines(inputLines, outputLabel, inImg, MM_PER_PIX):
    lines = Image.new("RGB", inImg.size, "white")
    draw = ImageDraw.Draw(lines)
    
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    for fooLine in inputLines:
        draw.line([(fooLine[0][1], fooLine[0][0]), (fooLine[1][1], fooLine[1][0])], fill="black", width=3)
        msp.add_line((fooLine[0][1]*MM_PER_PIX, - fooLine[0][0]*MM_PER_PIX), (fooLine[1][1]*MM_PER_PIX, - fooLine[1][0]*MM_PER_PIX), dxfattribs={"color": 2})
    
    lines.save(outputLabel+'.png')
    # lines.show()
    doc.saveas(outputLabel+'.dxf')