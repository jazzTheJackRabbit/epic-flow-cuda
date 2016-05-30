import os
import subprocess

base_path =  os.path.abspath("../data/other-data-full")+"/"
image_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

output = ""
for folder in image_folders:
	files = os.listdir(base_path + folder)
	folder_path = base_path + folder
	
	if not os.path.exists(folder_path+"/processed"):
		os.makedirs(folder_path+"/processed")

	for i in range(0,len(files)):
		if(i < len(files) - 1):
			ref_image = folder_path + "/" + files[i]
			next_image = folder_path + "/" + files[i+1]

			if(".png" in  ref_image and ".png" in next_image):
				# Call Run_all.sh from here
				base_frame_name = folder_path+"/processed/"+files[i][0:files[i].index(".png")]
				base_image_name = folder_path+"/processed/frame"
				subprocess.call(["./run_all.sh", ref_image, next_image, str(base_image_name + "-edge-file"), str(base_image_name + "-match-file"), str(base_image_name + "-flow.flo"),str(base_frame_name + ".png")])

	# subprocess.call(["./run_movie_maker.sh",str(folder_path+"/frame%02d.png"),str(folder_path+"/processed/original-movie.mp4")])
	# subprocess.call(["./run_movie_maker.sh",str(folder_path+"/processed/frame%02d.png"),str(folder_path+"/processed/flow-movie.mp4")])

	# subprocess.call(["./run_merge_movie.sh",str(folder_path+"/processed")])

	# output += "file "+folder_path+"/processed/output-final.mp4\n"	
# f = open(base_path+'final-demo-list.txt', 'w')
# f.write(output)
# f.close()
