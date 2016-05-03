#!/usr/bin/env python3
#Copyright (c) 2016, Sebastian Soyer

import os
import sys
import urllib
import tarfile
import associate
import itertools

freiburg1URL = "http://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"

#Reporthook to display download progress
def reporthook(blocknum, blocksize, totalsize):
    downloaded = min(blocknum * blocksize, totalsize)
    if totalsize > 0:
        #Downloading
        percent = downloaded * 100 / totalsize
        s = "%.1f%% %d/%d" % (
            percent, min(downloaded, totalsize), totalsize)
        print "\r",s,
        #Download finished -> add linebreak
        if downloaded >= totalsize:
            print ""
    else:
        #Unknown size
        sys.stderr.write("read %d\n" % (downloaded,))

#Download freiburg1_xyz dataset
os.system('setterm -cursor off')
print "Downloading benchmark dataset"
urllib.urlretrieve(freiburg1URL, 'rgbd_dataset_freiburg1_xyz.tgz', reporthook)
os.system('setterm -cursor on')

#Extract downloaded data
print "Extracting benchmark dataset"
try:
    tar = tarfile.open('rgbd_dataset_freiburg1_xyz.tgz', 'r:gz')
    for item in tar:
        tar.extract(item)
    print 'Done.'
except:
    print "Error opening downloaded file"

#Run assoc
print "Associating timestamps"
first_list = associate.read_file_list("rgbd_dataset_freiburg1_xyz/groundtruth.txt")
second_list = associate.read_file_list("rgbd_dataset_freiburg1_xyz/depth.txt")
third_list = associate.read_file_list("rgbd_dataset_freiburg1_xyz/rgb.txt")

#Match Groundtruth with depth
matchesDepth = associate.associate(first_list, second_list, 0.0, 0.02)

#Convert matched groundtruth-depth timestamps to full data tuples (gtTime, qx, qy, qz, qw, tx, ty, tz, depthTime, depthFile)
matchedDepthVals = []
for a,b in matchesDepth:
    matchedDepthVals.append([val for sublist in [[a], first_list[a], [b], second_list[b]] for val in sublist])

#Convert tuple to dict of form {gtTime: [qx, qy, qz, qw, tx, ty, tz], depthTime: [depthFile])
matchedDepthDict = {elem[0]:elem[1:] for elem in matchedDepthVals}
#Match created tuple with color times
matchesColor = associate.associate(matchedDepthDict, third_list, 0.0, 0.02)

#Convert matched groundtruth-depth-color timestamps to full data tuples (gtTime, qx, qy, qz, qw, tx, ty, tz, depthTime, depthFile, colorTime, colorFile)
matchedColorVals = []
for a,b in matchesColor:
    matchedColorVals.append([val for sublist in [[a], matchedDepthDict[a], [b], third_list[b]] for val in sublist])

#Write resulting list of tuples to file
print "Writing assoc file"
f = open('rgbd_dataset_freiburg1_xyz/rgbd_assoc_poses.txt', 'w')
for elem in matchedColorVals:
    f.write("%f %s %f %s %f %s\n"%(elem[0], " ".join(elem[1:8]), elem[8], elem[9], elem[10], elem[11]))
print "Done."
