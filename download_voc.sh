#!/bin/bash

# Create working directory
mkdir -p VOCdevkit

# Download VOC2012
echo "Downloading VOC2012 dataset..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo "Extracting VOC2012..."
tar -xf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar

# Download SBD augmentation labels
echo "Downloading SegmentationClassAug.zip from Dropbox..."
wget "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1" -O SegmentationClassAug.zip

echo "Downloading list.zip from Dropbox..."
wget "https://www.dropbox.com/s/4j4zv99q8mcz7f7/list.zip?dl=1" -O list.zip

echo "Unzipping files..."
unzip SegmentationClassAug.zip
unzip list.zip

# Move folders into VOC2012
echo "Moving folders to VOCdevkit/VOC2012..."
mv SegmentationClassAug VOCdevkit/VOC2012/
mv list VOCdevkit/VOC2012/

# Cleanup
rm SegmentationClassAug.zip
rm list.zip

echo "Download and setup completed."
