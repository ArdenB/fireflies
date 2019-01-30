# !/bin/bash

# ============================================================
# Bash script to allow for an uptodate conda log 
# ============================================================

filein="/home/ubuntu/Documents/fireflies/condapack/condapackagefile.txt"
# +++++ If an existing package list exists, move to old folder +++++ 

if [ -f $filein ]; then
	today=`date '+%Y_%m_%d__%H_%M_%S'`
	filename="/home/ubuntu/Documents/fireflies/condapack/old/condapackagefile$today.txt"
	# echo $filename;
	echo "Existing list found, moving to old folder"
	mv /home/ubuntu/Documents/fireflies/condapack/condapackagefile.txt $filename
	echo $filename
fi

# +++++ Export the conda package list +++++
conda list --explicit > $filein

# +++++ add it to the git repo +++++
git add $filein
git commit -am "Updating the conda environment list with new packages"
git push fire master