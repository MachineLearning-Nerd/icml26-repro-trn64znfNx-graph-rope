# Graph-RoPE / WIRE: CPU reproducibility certificate

This repository independently reproduces the three theoretical claims in
*Rotary Position Encodings for Graphs* (OpenReview `trn64znfNx`, arXiv
`2509.22259v4`). It is a theory certificate, not a substitute for the paper's
benchmark training runs. It runs on CPU with the Python standard library.

## Source provenance

- Primary artifact: arXiv v4 PDF, SHA-256
  `3eb6899ac0da995483dfba1eafe1ed625a1673d638d498882bcf741709f3415f`.
- The arXiv endpoint supplied a PDF rather than TeX; that limitation is
  explicit, and all theorem references are to that pinned PDF.
- Author release: `cederikhoefs/Graph-RoPE` at
  `4ac067eb38272543b0cdd7591d630399ff37bce4`.
- The verifier confirms the release rotates both `Q` and `K` before its
  `FastAttention` linear path.

## Reproduce

```bash
git submodule update --init --recursive
uv venv --python 3.12 .venv
source .venv/bin/activate
python repro/src/verify_graph_rope.py --output outputs/PUBLICATION_GATE_PASSED.json
python -m unittest repro.tests.test_graph_rope -v
```

## Claims and evidence

| Claim | Full certificate | Independent check and rejection control |
|---|---|---|
| C1 — WIRE recovers RoPE on grid graphs | Analytic path spectra for every `P_N`, `N=2..96`, and all node pairs on 121 Cartesian grids. It proves the Fiedler coordinate is monotone and checks the rotation composition identity. | Exact Laplacian residuals and orthogonality; rotating only queries changes the score and is rejected. The claim is stated accurately: recovery is RoPE under the theorem's bijective spectral-coordinate transformation, not literal integer positions. |
| C2 — randomized WIRE has leading effective-resistance dependence | All ordered pairs on `P_N` (`N=2..64`) and `C_N` (`N=3..64`) compare the spectral sum to independent electrical formulas. The Gaussian characteristic function gives the expectation and its fourth-order remainder; 100,000 seeded samples check the mean. | Omitting the required `1/sqrt(lambda)` weighting creates a large mismatch. This is an expectation/small-frequency theorem, not a claim about every learned frequency realization. |
| C3 — compatibility with linear attention | Direct token-pair linear attention equals the associative `phi(Q)ᵀ(phi(K)V)` computation after independent rotations across 372 full cases. | A pairwise relative-bias table requires `N²` entries, while the associated state is feature-by-value size. The author release's `GraphRoPE` → `FastAttention` path is source-attested. |

The generated JSON contains raw counts, tolerances, source pins, and all
control measurements. `repro/src/verify_graph_rope.py` intentionally does not
import the authors' model, which makes the numerical evidence independent of
their PyTorch implementation.

## Scope and cost

| | This reproduction | Full empirical replication |
|---|---|---|
| Scope | All three theoretical claims and the release's relevant code path | Training and benchmark tables from the paper |
| Hardware | CPU, standard Python | Task-dependent accelerators and datasets |
| Time | About 20 seconds for the certificate | Not run here |
| Cost | $0 | Not estimated |
| Outcome | Three claims passed with controls | Out of scope |

## Layout

- `repro/src/verify_graph_rope.py`: source-pinned, independent certificate.
- `repro/tests/`: regression tests.
- `outputs/PUBLICATION_GATE_PASSED.json`: generated gate artifact.
- `source/primary.pdf`: pinned primary artifact.
- `upstream/`: pinned author release.
