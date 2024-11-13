# Experiments with PLUMED 

A repository where I noodle around with PLUMED and the ASE plugin for PLUMED...

## Installation with `micromamba`

In order to install the environment using `micromamba`, you can use the `environment.yml` file: 

```bash
micromamba create -f environment.yml # first time
micromamba activate plumedenv
```
However, I found that on my local machine, I could not install `plumed` correctly in this manner (COLVAR file was created but the collective variable values were erroneously 0). Therefore, one can also install the environment using a `conda-lock` file like so:

```bash
micromamba create -n plumedenv -f conda-lock.yml
```

The multi-platform `conda-lock` file was generated from the `environment.yml` file using the eponymous [`conda-lock`](https://github.com/conda/conda-lock) :

```bash
conda-lock -f environment.yml -p osx-64 -p linux-64
```
Note: when installing `conda-lock` make sure to use `pipx`. 

Further, you need to install a python package to get some ASE extras I wrote

```bash
pip install -e ./python_package
```

## How to run on [Elja](https://irhpcwiki.hi.is/docs/intro/)

```bash
module load GCC/13.2.0
```
Make sure to load the GCC module before activating the environment. 

## Running the Snakemake with micromamba

In the top-level directory, first activate the micromamba environment
```bash
micromamba activate plumedenv
```

Note: On the cluster it is a good idea to run Snakemake in a `tmux` shell (because if it is killed then the jobs die). One might do the following: 

```bash
tmux new -s my_session
micromamba activate plumedenv
```
And then run Snakemake commands. To detach use <Ctrl-B then press D> and to re-attach

``` bash
tmux attach -t my_session
```

Regarding Snakemake: always do a dry run first, and use the `config.yml` inside `elja_profile`:
Go into the correct directory to run, for instance 1-dropH2O

```bash
snakemake -n --profile elja_profile/
```

**Caution:** Never run Snakemake on Elja without using the `config.yml` or else you will submit jobs to the login node.

To actually run the simulations, submit it like so: 

```bash
snakemake --profile elja_profile/
```

## Tutorials 

**`ase_plumed_tutorial`**- From the [ASE-PLUMED_tutorial](https://github.com/Sucerquia/ASE-PLUMED_tutorial) 