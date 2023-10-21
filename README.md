# Description

unRavel: Machine Learning Assisted RDP Bitmap Cache Forensics Tool

## Usage

Firstly, use ![bmc-tools](https://github.com/ANSSI-FR/bmc-tools) to extract the tiles from the cache:

```
mkdir inputs
python bmc-tools.py -s bcache24.bmc_Cache0000.bin -d inputs
```

Switch over to the `unravel` tool for the remaining steps.

Broadly, the steps are:
1. Separate the tiles according to their sizes
1. Extract the features from the images to generate the dataset
1. Generate diagrams to analyse the cluster.
1. Generate the prediction for k clusters.
1. Create the subfolders of images for every cluster in k.
1. Generate the collages for every cluster in k.

## Example: Running the tool on Windows

Separate the tiles accordinf to their sizes.
```
python unravel.py preprocess -i ..\input\raws -o ..\input\sizes
```

Clustering using colours
```
python unravel.py extract --colour_profile -i ..\input\sizes\64x64 -o ..\input\cp\dataset.csv

python unravel.py cluster --analyse -k 20 -i ..\input\cp\dataset.csv -o ..\input\cp

python unravel.py cluster --predict -k 7 -i ..\input\cp\dataset.csv -o ..\input\cp\prediction.csv

python unravel.py cluster --create -k 7 -i ..\input\cp\prediction.csv -o ..\input\cp\clusters

for /D %G in ("..\input\cp\clusters\*") do python unravel.py collage -i "%~fG" -o "%~fG-collage.jpg" -w 64
```

Clustering using contents
```
python unravel.py extract --vgg19 -i ..\input\cp\clusters\00 -o ..\input\cp\clusters\vgg19\00-dataset.csv

python unravel.py cluster --analyse -k 20 -i ..\input\cp\clusters\vgg19\00-dataset.csv -o ..\input\cp\clusters\vgg19\00-analysis\graph

python unravel.py cluster --predict -k 12 -i ..\input\cp\clusters\vgg19\00-dataset.csv -o ..\input\cp\clusters\vgg19\00-prediction.csv

python unravel.py cluster --create -k 12 -i ..\input\cp\clusters\vgg19\00-prediction.csv -o ..\input\cp\clusters\vgg19\00-clusters

for /D %G in ("..\input\cp\clusters\vgg19\00-clusters\*") do python unravel.py collage -i "%~fG" -o "%~fG-collage.jpg" -w 40


python unravel.py extract --vgg19 -i ..\input\cp\clusters\01 -o ..\input\cp\clusters\vgg19\01-dataset.csv

python unravel.py cluster --analyse -k 20 -i ..\input\cp\clusters\vgg19\01-dataset.csv -o ..\input\cp\clusters\vgg19\01-analysis\graph

python unravel.py cluster --predict -k 11 -i ..\input\cp\clusters\vgg19\01-dataset.csv -o ..\input\cp\clusters\vgg19\01-prediction.csv

python unravel.py cluster --create -k 11 -i ..\input\cp\clusters\vgg19\01-prediction.csv -o ..\input\cp\clusters\vgg19\01-clusters

for /D %G in ("..\input\cp\clusters\vgg19\01-clusters\*") do python unravel.py collage -i "%~fG" -o "%~fG-collage.jpg" -w 30
```

Check out the help message for more options.
```
python unravel.py --help
usage: unravel.py [-h] {preprocess,extract,cluster,collage} ...

unRavel: Machine Learning Assisted RDP Bitmap Cache Forensics Tool

positional arguments:
  {preprocess,extract,cluster,collage}
                        sub-commands help
    preprocess          Data preprocessing. Not all tiles have 64x64 dimension. Separate them into different pools
                        first.
    extract             Features extraction.
    cluster             K-Means clustering.
    collage             Generate collage of images.

options:
  -h, --help            show this help message and exit
```

## Troubleshooting

If the tool is run from Windows and it throws `OSError: [WinError 1314] A required privilege is not held by the client`,
run it as the administrator because administrator access is required to generate symlinks in Windows.
There is no such problem on Linux.
