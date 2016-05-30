import os
import subprocess

base_path = "../data/other-data/"
image_folders = os.listdir(base_path)

for folder in image_folders:
	files = os.listdir(base_path + folder)	
	folder_path = base_path + folder
	
	for file in files:
		print(file)
		image = folder_path+"/frame"
		starting_frame = 7
		ext = "png"

		# Call Run_all.sh from here
		subprocess.call(["./run_all.sh", ref_image, next_image, str(folder_path + "/frame10" + "-edge-file"), str(folder_path + "/frame10" + "-match-file"), str(folder_path + "/frame10" + ".flo"),str(folder_path + "/frame10-11" + ".png")])

	break