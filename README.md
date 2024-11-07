# Experiments with PLUMED 

A repository where I noodle around with PLUMED and the ASE plugin for PLUMED...

## How to run on Elja

```bash
module load GCC/13.2.0
```

Install with the micromamba environment using a conda-lock file like so: 

```bash
micromamba create -n plumedenv -f conda-linux-64.lock
```

But the lock file was generated with a different Python version. If you want GPAW, this needs a different Python version. To get around this, run the following: 

```bash
micromamba install -c conda-forge python==3.12 gpaw
```