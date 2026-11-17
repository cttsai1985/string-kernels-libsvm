# Performance benchmarks

This page records the v1.0.1 algorithm/data-reuse measurements. For v1.1.0
SIMD and multicore measurements and how to optimize speed, see the
[dedicated optimization guide](OPTIMIZATION.md).

These benchmarks measure kernel-matrix construction, not model accuracy or SVM
training. The `1.0.1` refactor preserves the published kernels and their
normalization. Weighted-degree pairs now use a linear scan of matching runs;
spectrum matrices prepare each sequence's feature counts once; Python training
matrices evaluate one triangle and mirror it. Evaluation uses one native thread.

## Measured results

**MEASURED on 2026-11-17:** AMD Ryzen Threadripper PRO 5965WX, Linux/WSL2 x86_64,
CPython 3.12.13, NumPy 2.2.6. Both project versions were built locally with GCC
13.3.0, CMake 4.4.4 Release (`-O3 -DNDEBUG`), pybind11 3.1.0 and its enabled LTO.
No architecture-specific optimization flags were added. The baseline source is
[commit 81f5e6b](https://github.com/cttsai1985/string-kernels-libsvm/commit/81f5e6b376e144635ea223beb373c780d10be954),
which has the `1.0.0` implementation. The candidate is the issue #22 patch shipped
as `1.0.1`. Binary hashes, build descriptions, input hashes, CPU/wall times,
thread-pool information, and every trial are retained in
[the raw results](../benchmarks/results/linux.json).

Times below are **MEASURED milliseconds: median [minimum–maximum]**, across five
fresh-process trials per backend and case. Dimensions list rows × columns,
sequence length, and degree (exact k-mer size for spectrum).

| Workload | Dimensions; length; degree | v1.0.0 | v1.0.1 | strkernels, 1 thread | strkernels, 4 threads |
| --- | --- | --- | --- | --- | --- |
| Tiny WD prediction | 16 × 12; 12; 3 | 0.021 [0.020–0.025] | 0.020 [0.020–0.020] | 0.096 [0.092–0.124] | 0.093 [0.091–0.098] |
| Protein WD training | 192 × 192; 64; 5 | 2.916 [2.904–2.976] | 1.370 [1.355–1.454] | 8.927 [8.861–8.967] | 3.832 [3.662–3.980] |
| Highly similar WD training | 48 × 48; 192; 64 | 12.606 [12.527–12.738] | 0.231 [0.228–0.251] | 95.708 [94.523–100.180] | 57.151 [52.840–66.933] |
| Protein WD prediction | 128 × 256; 64; 5 | 2.674 [2.662–2.686] | 2.474 [2.414–2.709] | 15.554 [15.400–15.726] | 4.702 [4.433–5.237] |
| Degree-one WD training | 512 × 512; 32; 1 | 21.483 [21.388–23.070] | 11.916 [11.737–12.313] | 18.396 [18.323–18.477] | 6.457 [6.372–6.744] |
| Exact-3 spectrum training | 128 × 128; 128; 3 | 89.833 [88.650–93.088] | 8.183 [8.092–9.226] | Excluded | Excluded |

The largest improvement occurs on long matching runs at high degree: the old
algorithm revisited each matching prefix while the new scan sums its weights in
closed form. **MEASURED** baseline native profiling with py-spy 0.4.1 at 20 Hz
collected 159 samples without errors; 155 had `weighted_degree` as the leaf.
The profile used 128 similar length-256 sequences at degree 128, repeatedly
computing complete matrices. The
[Speedscope profile](../benchmarks/results/weighted-degree-baseline.speedscope.json)
is retained for inspection. Spectrum gains come from retaining feature counts
between pair evaluations within one matrix call, combined with symmetry reuse.

The small and rectangular cases gain much less. Four-thread `strkernels` wins
the degree-one workload. This suite does not establish a universally fastest
implementation, biological predictive quality, or full training-pipeline speedup.
It covers deterministic synthetic workloads; representative application data may
have different length, similarity, feature diversity, and memory requirements.

### Standalone converter

These **MEASURED milliseconds** include process startup, input parsing, kernel
preparation, and LIBSVM text output to a temporary file. Reading the generated
matrix back for comparison is outside the timed region. Both CLI executables
were built with GCC 13.3.0 and the same CMake Release flags. They stream complete
rows rather than storing a dense matrix or reusing its opposite triangle.

| Workload | v1.0.0 CLI | v1.0.1 CLI |
| --- | --- | --- |
| Tiny WD prediction | 2.447 [2.207–2.725] | 2.407 [2.304–2.478] |
| Protein WD training | 16.310 [15.899–16.822] | 16.187 [16.061–17.064] |
| Highly similar WD training | 16.094 [15.975–16.895] | 3.476 [3.355–3.738] |
| Protein WD prediction | 15.736 [15.439–20.058] | 15.474 [15.252–15.797] |
| Degree-one WD training | 94.257 [90.071–98.295] | 95.379 [91.583–108.008] |
| Exact-3 spectrum training | 106.346 [105.526–109.696] | 24.805 [24.127–26.018] |

CLI gains differ from Python because these are different operation boundaries.
Some medians barely change, and the degree-one CLI median is slightly slower
with overlapping trial ranges. Numeric serialization and startup remain included;
these results do not isolate their individual costs.

## Method and limits

The harness fixes seed `20261117`, constructs identical inputs for every backend,
and verifies **every matrix cell** against the baseline before accepting timing
results (`rtol=1e-11`, `atol=1e-12`). Independent feature-map tests also verify the
mathematics; agreement between implementations alone is insufficient. Training
cases pass the same array as both axes so every backend can exploit symmetry.
Prediction cases have separate ordered reference and target cohorts.

Each worker imports only its selected implementation. One untimed warmup is
followed by adaptive timed blocks of at least 0.2 seconds for this report. Blocks
include input conversion, per-call preparation, allocation, normalization, and
matrix evaluation. No matrix-result cache is used. Imports are excluded, while
first-call warmup times are retained separately. Backend order is shuffled within
each trial. Trials run serially, with no concurrent project builds or tests.

The process was restricted to logical CPUs `0,2,4,6`, representing four distinct
physical cores. Native project code is single-threaded. `strkernels` was tested
with `OMP_NUM_THREADS=1` and `4`, `OMP_DYNAMIC=FALSE`; BLAS thread counts were
set to one. CPU/wall ratios and discovered OpenMP pools are in the raw results.
Affinity and thread counts do not guarantee exclusive hardware or eliminate
operating-system noise; the trial ranges expose some of that variation.

[strkernels 0.2.15](https://pypi.org/project/strkernels/0.2.15/) is Python over a
compiled C core. Its [build configuration](https://github.com/denilsonfbar/strkernels/blob/main/setup.py)
enables OpenMP, and its
[matrix implementation](https://github.com/denilsonfbar/strkernels/blob/main/strkernels/core/str_kernel_matrix.c)
reuses symmetry and parallelizes rows. Its normalized weighted-degree definition
matches these tested inputs, despite a constant raw-weight scaling difference.
Its spectrum method sums orders up to the requested order; this project uses
exact-k counts. That cross-library spectrum comparison is excluded. `strkernels`
is an optional GPL-3.0 benchmark dependency and is not linked into project wheels.

Python output storage remains analytical `8*N*M` bytes for float64 values, plus
inputs and preparation. Cached spectrum maps add storage proportional to the
number of distinct k-mers across sequences. The CLI retains only prepared
sequence state and the current output row. No peak-RSS measurement was collected;
these are storage descriptions, not measured memory comparisons. SSK and
positional-kernel performance were not benchmarked in this release.

## Reproduce the measurement

Use an environment with a C++17 compiler, Git, Python development headers, and
the pinned test/benchmark extras. Build tools are fixed in `pyproject.toml`.
From a clone containing the baseline commit:

```sh
python -m pip install '.[test,benchmark]' build==1.6.1
```

Prepare and build both source versions with the same interpreter and compiler.
The following Python commands avoid shell-specific archive/extraction commands:

```python
from pathlib import Path
import io, subprocess, sys, tarfile, zipfile

root = Path('build/benchmark-reproduction')
root.mkdir(parents=True, exist_ok=True)
source = root / 'baseline-source'
source.mkdir(exist_ok=True)
commit = '81f5e6b376e144635ea223beb373c780d10be954'
data = subprocess.check_output(['git', 'archive', commit])
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    archive.extractall(source, filter='data')  # Python 3.12+
for name, path in [('baseline', source), ('candidate', Path('.'))]:
    dist = root / f'{name}-dist'
    subprocess.run([sys.executable, '-m', 'build', str(path), '--wheel',
                    '--outdir', str(dist.resolve())], check=True)
    with zipfile.ZipFile(next(dist.glob('*.whl'))) as archive:
        archive.extractall(root / f'{name}-package')
```

Then run the installed-artifact comparison:

```sh
python benchmarks/benchmark_kernels.py --baseline-root build/benchmark-reproduction/baseline-package --candidate-root build/benchmark-reproduction/candidate-package --include-strkernels --trials 5 --seconds 0.2 --output build/benchmark-reproduction/results.json
```

Optional `--baseline-cli` and `--candidate-cli` accept Release executables built
from those same sources using the [development instructions](DEVELOPMENT.md).
`--provenance build-description.json` embeds compiler flags and source descriptions
with the output. Linux affinity defaults to four distinct physical cores when
available; `--affinity-width 1` supports a single-core run. Platforms without
`sched_setaffinity` retain their default scheduler affinity. Run comparisons on
an otherwise idle machine and record the environment when sharing results.

[The CI benchmark workflow](../.github/workflows/benchmarks.yml) builds both source
versions with the same toolchain, compares complete matrices, and uploads raw
results. Its deliberately broad high-similarity floor requires a threefold
improvement over the baseline; the local measured gap is much larger. Comparing
the baseline to itself failed this gate before optimization (MEASURED ratio
1.082). Unit tests also demonstrate that wrong off-diagonal values, wrong shapes,
nonfinite matrices, and insufficient speedups are rejected. Hosted timing is
noisier than a dedicated benchmark host; this gate detects loss of the main
algorithmic improvement rather than promising a precise latency.
