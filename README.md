# From Prediction to Pressure
[![DOI](https://zenodo.org/badge/1248935194.svg)](https://doi.org/10.5281/zenodo.20677241)


Code, data, and the manuscript for the paper *"Wrong with Confidence: How Large
Language Models Hallucinate, Manipulate, and Discriminate."*

This is the empirical companion to our earlier theoretical work on the structural
origins of hallucination in large language models (arXiv:2606.07537). The theory
paper argues that hallucination is a consequence of how transformers are built
rather than an accident of their data. This repository contains the experiment we
ran to test that argument, end to end, so that anyone can check the numbers for
themselves.

We pre-registered the study before scoring any output. Five open-weight models were
run under fixed decoding settings on a six-condition prompt set and three identity
probes, giving 5,620 responses. Two of us scored every response independently and
blind to the model. The full analysis is reproducible from the files here.

## Layout

```
code/      scripts to sample questions, collect outputs, score reliability, and analyse
data/      prompts, identity probes, raw model outputs, and the scored sheets
results/   the kappa report and the figures used in the paper
paper/     the LaTeX source and compiled PDF (IEEE conference format)
docs/      pre-registration, scoring rubric, and the prompt taxonomy
```

## Reproducing the analysis

The downstream analysis runs from the scored sheets that are already in `data/`.
From the repository root:

```bash
pip install -r requirements.txt

python code/compute_kappa.py     # inter-annotator agreement (Cohen's kappa)
python code/run_analysis.py      # all hypothesis tests, with FDR correction
python code/make_figures.py      # regenerates every figure into results/figures/
```

`compute_kappa.py` and `run_analysis.py` print the exact numbers reported in the
paper. `make_figures.py` rebuilds the figures from the same data.

## Collecting outputs from scratch

`code/sample_benchmarks.py` draws the question sets and `code/collect_api_data.py`
queries the models. These need access to the model endpoint and the benchmark
sources, and they write into `data/raw_outputs/`. The raw outputs we used are
already included, so you only need these if you want to repeat the collection on
other models. Decoding settings are fixed at the top of the collection script.

## Data notes

- Model identity and the clear/ambiguous label are hidden in the scoring sheets and
  kept separately in `blind_key.csv`; this is what kept the scoring blind.
- `kappa_overlap_uids.csv` lists the 600 responses both annotators scored, which is
  the basis for the agreement numbers.
- The confidence-tone column was recalibrated once, to the pre-registered
  definition, after the first agreement check; this is noted in the paper and the
  pre-registration.

## Citation

If you use this work, please cite the paper and the companion framework:

```
M. R. K. Sadi, T. R. Tasin, and D. F. Chowdhury, "Wrong with Confidence: How Large
Language Models Hallucinate, Manipulate, and Discriminate," 2026.

M. R. K. Sadi, T. R. Tasin, and G. M. Naeem, "From Architecture to Output:
Structural Origins of Hallucination in Large Language Models and the Amplifying
Role of Data," arXiv:2606.07537, 2026.
```

## License

Code is released under the MIT License (see `LICENSE`). The data and figures are
released under CC BY 4.0.

---

Maintained by Md. Rejaul Korim Sadi, Department of Computer Science and Engineering,
Metropolitan University, Sylhet, Bangladesh.
