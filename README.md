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

## How to run on [Elja](https://irhpcwiki.hi.is/docs/intro/)

```bash
module load GCC/13.2.0
```
Make sure to load the GCC module before activating the environment. 

## Tutorials 

**`ase_plumed_tutorial`**- From the [ASE-PLUMED_tutorial](https://github.com/Sucerquia/ASE-PLUMED_tutorial) 