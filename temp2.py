import os
import shutil

folder = "D:/A_my_study/visdrone/val/daytime/labels"
label = "D:/A_my_study/visdrone/val/vallabelr/"

dest_label = "D:/A_my_study/visdrone/val/daytime/labelsr"

for filename in os.listdir(folder):
    # name, ext = os.path.splitext(filename)
    # label1 = label + name + ".xml"
    label1 = label + filename
    if os.path.exists(label1):
        shutil.move(label1,dest_label)