# Optimal Brain Compression

## Source: github.com/IST-DASLab/OBC

## Files

* `trueobs.py`: efficient implementations of ExactOBS for all compression types
* `main_trueobs.py`: code to run ExactOBS 
* `post_proc.py`: post processing operations like statistics corrections
* `database.py`: generating databases for non-uniform compression
* `spdy.py`: implementation of the DP algorithm for finding non-uniform
  compression configurations; adapted from code provided by the authors of SPDY [9]
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantization utilities

## Usage 

First, make sure ImageNet is located/linked to `../imagenet` (alternatively,
you can specifiy the `--datapath` argument for all commands).