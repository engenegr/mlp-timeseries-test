### Dataset insight

It is possible to build simple HTML page to make understanding of the available
database easier. 

For that, use [Flenser](https://github.com/JohnMcCambridge/flenser) tool.

```shell
python flenser.py <PATH_TO_DATASET_CSV>
```

will produce output HTML file.

### Using workflow

1. Install dependencies from `requirements-3.10.txt` in `virtualenv`. Expected Python version is 3.10.

2. Run workflow with `snakemake -j1`. It downloads csv file and saves normalized arrays for model training.

3. Generate HTML report, optionally `snakemake --report`

