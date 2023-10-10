#!/usr/bin/env python3
import argparse
from helpers import *


def main(command_line=None):
    parser = argparse.ArgumentParser(
        description = 'unRavel: Machine Learning Assisted RDP Bitmap Cache Forensics Tool'
    )

    subparsers = parser.add_subparsers(
        dest = 'command',
        help = 'sub-commands help',
        required = True
    )

    # Data Preprocessing options
    preprocess = subparsers.add_parser(
        'preprocess',
        help = 'Data preprocessing. Not all tiles have 64x64 dimension. Separate them into different pools first.'
    )
    preprocess.add_argument(
        '-i',
        '--input_folder',
        help = 'The input folder of the bitmap files that are extracted from the cache.',
        required = True
    )
    preprocess.add_argument(
        '-o',
        '--output_folder',
        help = 'The output folder where the pools will be created.',
        required = True
    )

    # Feature Extraction options
    extract = subparsers.add_parser(
       'extract',
        help = 'Features extraction.'
    )
    extract_group = extract.add_mutually_exclusive_group(required=True)
    extract_group.add_argument(
        '-cp',
        '--colour_profile',
        help = 'Extract the colour profiles of the images. Images are quantized to 256 colours.',
        action = 'store_true'
    )
    extract_group.add_argument(
        '--hog',
        help = 'Extract features using the Histogram of Oriented Gradients.',
        action = 'store_true'
    )
    extract_group.add_argument(
        '--prewitt',
        help = 'Detect edges using the Prewitt operator.',
        action = 'store_true'
    )
    extract_group.add_argument(
        '--vgg16',
        help = 'Extract features using the VGG16 object detection algorithm.',
        action = 'store_true'
    )
    extract_group.add_argument(
        '--vgg19',
        help = 'Extract features using the VGG19 object detection algorithm.',
        action = 'store_true'
    )
    extract.add_argument(
        '-i',
        '--input_folder',
        help = 'The input folder of the pool of bitmap files that have been preprocessed, e.g. the 64x64 folder.',
        required = True
    )
    extract.add_argument(
        '-o',
        '--output_csv',
        help = 'The dataset that will be generated.',
        required = True
    )

    # Clustering options
    cluster = subparsers.add_parser(
        'cluster',
        help = 'K-Means clustering.'
    )
    cluster_group = cluster.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument(
        '-a',
        '--analyse',
        help = 'Generate a graphs to determine what is the optimal number of clusters, given the maximum number (k) of clusters.',
        action = 'store_true'
    )
    cluster_group.add_argument(
        '-p',
        '--predict',
        help = 'Train the k-means model and cluster the input into the specified k number of clusters.',
        action = 'store_true'
    )
    cluster_group.add_argument(
        '-c',
        '--create',
        help = 'Create folders to store clusters of images.',
        action = 'store_true'
    )
    cluster.add_argument(
        '-k',
        type = int,
        help = 'k is the number of clusters to train the k-means clustering algorithm.',
        required = True
    )
    cluster.add_argument(
        '-i',
        '--input_csv',
        help = 'The input dataset that contains the features extracted from the images.',
        required = True
    )
    cluster.add_argument(
        '-o',
        '--output',
        help = '''The output file name.
If you are using the analyse option, the output is an image of the plot.
If you are using the predict option, the output is a file that contains a dataset of the features and the prediction.
If you are using the create option, the output is a parent folder that will store the subfolders of clusters.''',
        required = True
    )
    cluster.add_argument(
        '--pca',
        help = 'Specify the number of principal components for Principal Component Analysis. Default value is 50.',
        type = int,
        required = False,
        default = 50
    )

    # Collage options
    collage = subparsers.add_parser(
        'collage',
        help = 'Generate collage of images.'
    )
    collage.add_argument(
        '-i',
        '--input_folder',
        help = 'The folder that contains the images.',
        required = True
    )
    collage.add_argument(
        '-o',
        '--output_file',
        help = 'The collage will be stored as output_file.',
        required = True
    )
    collage.add_argument(
        '-w',
        '--width',
        help = 'The number of tiles per row. Default: 16.',
        required = True,
        type = int,
        default = 16
    )

    args = parser.parse_args(command_line)
    print (args.command)

    if args.command == 'preprocess':
        preprocess = Preprocess()
        preprocess.start(args.input_folder, args.output_folder)
    elif args.command == 'extract':
        # create the output folder if it doesn't exist
        output_folder = os.path.dirname(args.output_csv)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if args.colour_profile:
            algo = ColourProfiles()
        elif args.hog:
            algo = Contents_Hog()
        elif args.prewitt:
            algo = Contents_Prewitt()
        elif args.vgg16:
            algo = Contents_VGG16()
        elif args.vgg19:
            algo = Contents_VGG19()
        algo.generate_dataset(args.input_folder, args.output_csv)
    elif args.command == 'cluster':
        if args.pca < 0:
            print ("[-] PCA value must be positive.")
        else:
            cluster = Cluster(args.input_csv, args.output, args.pca)
            if args.analyse:
                #cluster.analyse(args.k)
                # TODO drop WCSS
                cluster.analyse_silhouette(args.k)
            elif args.predict:
                cluster.predict(args.k)
            elif args.create:
                cluster.create(args.k)
    elif args.command == 'collage':
        collage = Collage()
        collage.make(args.input_folder, args.output_file, args.width)


if __name__=='__main__':
    main()

