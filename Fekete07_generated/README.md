# Fekete et al. (2007), Section 6.2: newly generated OKP-3 instances

These are **new random draws**, not the original instances used to produce the
paper's results. The paper does not identify its random seeds. They follow its
Table 5 distributions and Section 6.2 procedure: 100 x 100 x 100 container,
three instance classes, five (number of box types, multiplicity) combinations,
and ten instances per combination: 150 files in total. Box dimensions are sampled
uniformly from the published intervals and rounded up; each box type receives an
independent class and an independent uniform integer profit multiplier in 1..3.

The seed is 2007. Filenames encode class, type count, multiplicity and replicate.
Each BOXES row holds type ID, multiplicity, three dimensions and the **full
profit**; use `--value-mode flat` with `3DMHKP-SA.py`. The paper's original
problem assumes fixed orientations, whereas that solver allows rotation, so
its results are for a relaxation and are not directly comparable.

To regenerate in a fresh directory, run `generate_fekete07.py` with an
alternative `--output-dir` and optionally `--seed`. The generator refuses to
overwrite existing instance files. It uses only Python's standard library.