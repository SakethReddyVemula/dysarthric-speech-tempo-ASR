 1017  clear
 1018  mfa
 1019  clear
 1020  conda create -n aligner -c conda-forge montreal-forced-aligner
 1021  conda activate aligner
 1022  python3 force_align.py 
 1023  clear
 1024  ls
 1025  clear
 1026  ls
 1027  conda --list
 1028  clear
 1029  pip3 freeze
 1030  conda deactivate
 1031  ls
 1032  clear
 1033  pip3 freeze
 1034  clear
 1035  conda activate aligner
 1036  conda config -add channels conda-forge
 1037  conda config --add channels conda-forge
 1038  clear
 1039  python3 force_align.py 
 1040  clear
 1041  python3 force_align.py -c conda-forge
 1042  history
 1043  conda activate mfa
 1044  python3 force_align.py 
 1045  conda deactivate


# Steps to run the codes

1. Activate the conda environment mfa:
```
conda activate mfa
```

2. Run the codes