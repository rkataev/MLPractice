import cv2
import struct
import gzip as gz
import numpy as np

filePath = '../MNISTdata/'
imgString = '../raw_imgs/image'
imgStringFormat = '.png'
tempArr = np.zeros((28, 28))

requestedImgNum = int(input('How many images to unpack? Enter "-1" to unpack all: '))

with gz.open('../MNISTdata/train-images-idx3-ubyte.gz', 'rb') as lscan:
    magicNum = lscan.readline(4)
    imgNum = struct.unpack('>i', lscan.read(4))
    rowsNum = struct.unpack('>i', lscan.read(4))
    colsNum = struct.unpack('>i', lscan.read(4))

    if requestedImgNum == -1:
        iterNum = imgNum[0]
    else:
        iterNum = requestedImgNum

    for imgIter in range(0, iterNum):
        print('Now working image: ' + str(imgIter))
        for x in range(0, rowsNum[0]):
            for y in range(0, colsNum[0]):
                tempArr[x][y] = struct.unpack('>B', lscan.read(1))[0]
                cv2.imwrite(imgString+str(imgIter)+imgStringFormat, tempArr)