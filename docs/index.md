# Artificial Neural Networks and Deep Learning

???+ info inline end "Edition"

    2026.2 — Insper

**Henrique Turco** — [github.com/henriquetg1](https://github.com/henriquetg1)

This site is my portfolio for the course
[Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/2026.2/)
(Insper, 2026.2). Each exercise lives in its own folder under `docs/exercises/`,
with the report (`index.md`), the code that was actually run (`code/`), and the
figures shown in the report (`figures/`).

## Deliverables

- [x] [1. Data](exercises/data/index.md) — 10.sep.2026
- [ ] 2. Perceptron
- [ ] 3. MLP
- [ ] 4. VAE
- [ ] Projects

## Reproducing

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python docs/exercises/data/code/main.py   # regenerates figures, tables and results.json
mkdocs serve -o                           # preview the site locally
```
