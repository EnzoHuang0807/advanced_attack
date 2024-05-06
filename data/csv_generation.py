import os
import csv

img_list = os.listdir("images")
with open("labels.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])
    for img in img_list:
        writer.writerow([img, int(img.split("_")[0])])
