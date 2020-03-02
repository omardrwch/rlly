import os
import shutil
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(dir_path)

dir_to_copy = os.path.join(project_path, 'rlly')
dir_destination = os.path.join(dir_path, 'all_files')

if not os.path.exists(dir_destination):
    os.makedirs(dir_destination)

# Copy all project files (.cpp and .h) from dir_to_copy to dir_destination
for root, dirs, files in os.walk(dir_to_copy):  
    for file in files:
        path_file = os.path.join(root,file)
        shutil.copy2(path_file,dir_destination) 


"""

Create header file to be used by acme.py

"""

header_contents = "#ifndef __RLLY_ENVS_H__ \n#define __RLLY_ENVS_H__ \n"

# List all source files
source_dir = dir_destination
source_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(source_dir):
    for filename in f:
        if '.h' in filename and filename != "rlly.hpp":
            print(filename)
            header_contents += "#include " + "\""  + filename + "\"" + "\n"
            
for r, d, f in os.walk(source_dir):
    for filename in f:
        if '.cpp' in filename:
            print(filename)
            header_contents += "#include " + "\""  + filename + "\"" + "\n"

header_contents += "\n #endif"
header_file = open(os.path.join(dir_destination, "rlly.hpp"),"w+")
header_file.write(header_contents)
header_file.close()