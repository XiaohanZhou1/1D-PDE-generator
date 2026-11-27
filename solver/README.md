# twstab

Spectral-stability labels for travelling waves of one-dimensional partial differential equations (PDEs).

- **Input:** a list of PDEs written with SymPy. Each entry holds the equations and the known conditions (parameter values; end states or wave speed where the class needs them).
- **Output:** a list. For each PDE it gives the travelling waves and one label for each wave: `STABLE`, `UNSTABLE` or `UNCERTAIN`.

The solver does every step by itself: it finds the constant states, computes the waves as boundary-value problems, linearises about each wave, and locates the spectrum (essential spectrum from the dispersion relations, eigenvalues from an Evans-function determinant with an independent finite-difference check). The results are numerical conclusions, not proofs.

Version 2.18.0. Tested with Python 3.12, NumPy 2.5, SciPy 1.18, SymPy 1.14.

---

## 1. Covered PDE types

A travelling wave is a solution `u(x, t) = U(x - c t)` that tends to constant states as `x -> -inf` and `x -> +inf`. The speed `c` can be zero. Three kinds are covered: a **front** (two different end states), a **pulse** (the same state at both ends) and, for free-boundary problems, a **semi-wave** (a constant state far behind the moving boundary).

All equations are in one space dimension, with coefficients that do not depend on `x` or `t` explicitly. Every parameter must have a numerical value.

| # | Type | Examples | What an entry must give |
|---|---|---|---|
| 1 | Reaction-diffusion equations and systems | Nagumo (bistable), Fisher-KPP, Allen-Cahn | equations, parameter values |
| 2 | Systems with components that do not diffuse (slow-fast, partly parabolic) | FitzHugh-Nagumo | equations, parameter values |
| 3 | Non-self-adjoint systems; cross-diffusion (not degenerate) | advection terms, coupled diffusion matrices | equations, parameter values |
| 4 | Nonlocal equations: convolution with a kernel, with or without diffusion | nonlocal Allen-Cahn, neural field | equations, parameter values |
| 5 | Free boundary with a Stefan condition (Dirichlet, Neumann or Robin condition) | Fisher-Stefan | equations, `free_boundary`, parameter values |
| 6 | Time delays: one or several discrete delays, distributed delays | delayed Nagumo | equations, parameter values |
| 7 | Parabolic equations of order 4 and 6 | extended Fisher-Kolmogorov and its sixth-order analogue | equations, parameter values |
| 8 | Second order in time, with or without damping | hyperbolic Nagumo, phi-4, Klein-Gordon | equations, parameter values |
| 9 | Transport towards a component that does not diffuse (taxis) | haptotaxis, chemotaxis with an immobile attractant | equations, parameter values |
| 10 | Equations with a rotation symmetry of two components | Ginzburg-Landau type, nonlinear Schrodinger type | equations, parameter values (see note c) |
| 11 | Conservation laws: scalar, systems, higher order | Burgers, p-system, Keller-Segel shock, thin film, Cahn-Hilliard | equations, parameter values, **end states** |
| 12 | Dispersive equations (solitary waves) | generalised Korteweg-de Vries | equations, parameter values, **rest state and speed** |

Notes.

- (a) **Types 1-10 need the equations and the parameter values only.** The solver searches for the waves and lists every wave that it finds. You can also name one wave by its end states (section 2.3).
- (b) **Types 11 and 12 need more.** In a conservation law every constant is an equilibrium, so the equation alone does not select the end states. A dispersive solitary wave exists for a family of speeds, so the speed must be given.
- (c) **Travelling waves only.** For type 10 the solver lists the waves with frequency zero. Rotating waves `exp(i w t) U(x - c t)` with `w != 0` (for example the solitons of the nonlinear Schrodinger equation) are not travelling waves and are not listed. Write a complex field as two real components.
- (d) **Multi-pulses.** For a stable pulse with an oscillatory tail the solver also lists the double pulses at the three smallest distances.

The travelling-wave backend does not cover degenerate diffusion (porous-medium type); taxis combined with a free boundary; periodic wave trains; waves whose end states are not constant; equations that contain `x` or `t` explicitly; more than one space dimension; delays that depend on the state.

The generator repository adds a separate `twstab floquet` command for applying the one-period map of its sampled F2 calibration orbit. [Temporal Floquet input and output](../docs/twstab-floquet.md)

---

## 2. Input format

The input is a **list**. Each entry describes one PDE. In a file, the list must have the name `problems`.

```python
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)                      # every unknown is a function of (x, t), in this order
a = sp.Symbol("a", positive=True)

eqn1 = dict(equations=sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)),
            parameters={a: 0.25})

eqn2 = dict(equations=sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2)),
            wave=dict(kind="front", left={u: 2}, right={u: 0}))

problems = [eqn1, eqn2]                         # ... up to eqn100, or any length
```

Entry `k` gets the name `eqn<k>` (`eqn1`, `eqn2`, ...) unless it gives its own `name`.

### 2.1 Keys of an entry

| Key | Content | Required |
|---|---|---|
| `equations` | one `sympy.Eq`, or a list of `sympy.Eq` for a system | yes |
| `parameters` | `{symbol: number}`; every parameter of the equations needs a value | if the equations contain parameters |
| `wave` | the known conditions on the wave (section 2.3) | types 11 and 12; optional otherwise |
| `free_boundary` | conditions at the moving boundary (section 2.2) | type 5 |
| `name` | name of the entry in the output | no |
| `unknowns` | order of the components, e.g. `[u, w]` | only with constraint equations |
| `options` | controls, e.g. `dict(equilibria_range=3.0)` (section 2.4) | no |
| `x`, `t` | the independent variables | no: they are read from the equations |

An entry without parameters and without `wave` can be the `sympy.Eq` itself, or a list of `sympy.Eq`:

```python
problems = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))]
```

### 2.2 How to write the equations

- Write each equation as `sp.Eq(left, right)`. It must contain a first or second time derivative of an unknown, except for constraint equations (below).
- **System:** a list of equations, one for each unknown.
- **Conservation form:** write the flux derivative as it is, e.g. `(u**2 / 2).diff(x)`.
- **Second order in time:** `u.diff(t, 2)`.
- **Time delay:** use the function without arguments, `u = sp.Function("u")`, and write `u(x, t - tau)`. A distributed delay is `sp.Integral(g(s) * u(x, t - s), (s, 0, sp.oo))`.
- **Convolution:** `sp.Integral(K * u(y, t), (y, -sp.oo, sp.oo))`, where the kernel `K` is an expression in `x - y`.
- **Constraint equation** (a nonlocal term written as an auxiliary unknown): add an equation without a time derivative, e.g. `sp.Eq(0, w.diff(x, 2) - w + u)`, and give `unknowns=[u, w]`.
- **Free boundary:** the domain is `x < s(t)`. Give

  ```python
  free_boundary=dict(conditions=[sp.Eq(u, 0)],        # conditions at x = s(t)
                     speed=-kappa * u.diff(x))        # Stefan condition: ds/dt = speed
  ```

### 2.3 The key `wave`: known conditions on the wave

| Key | Content |
|---|---|
| `kind` | `"front"`, `"pulse"`, or `"semi"` (free boundary) |
| `left`, `right` | end states of a front at `x -> -inf` and `x -> +inf`, e.g. `left={u: 2}, right={u: 0}` |
| `rest` | the end state of a pulse (both ends) |
| `far` | the state of a semi-wave far behind the free boundary |
| `speed` | the wave speed, if it is known. Required for type 12. Leave it out otherwise: the solver computes it |

Rules.

- **No `wave`:** the solver searches for the waves of the equation and labels each wave that it finds.
- **With `wave`:** the solver computes the wave with these end states and labels it. No guess of the profile is needed. If the search finds several waves that satisfy the conditions (for example a pulse and its mirror image), each of them is listed.
- **Conservation laws (type 11):** `kind`, `left` and `right` are required. The speed follows from the jump condition; the end states must satisfy it.
- **Dispersive solitary waves (type 12):** `kind="pulse"`, `rest` and `speed` are required.

Optional hints select one particular wave or help a difficult case: `guess={u: expression in sp.Symbol("z")}`, `amplitude={u: number}`, `width=number`, `speed_guess=number`.

### 2.4 Options

| Option | Meaning | Default |
|---|---|---|
| `equilibria_range` | constant states are searched in `[-range, range]` for each component | 100 |
| `double_pulses` | number of double pulses that are listed; 0 means none | 3 |
| `region` | `(x_max, y_max)`: rectangle of the eigenvalue search, if you want to set it | computed |

---

## 3. Output

The output is a **list with one entry for each PDE, in the order of the input**.

```python
[
  {"name": "eqn1",
   "waves": [
     {"wave": "front (u = 0) -> (u = 1)", "speed": -0.353553, "label": "STABLE", "qualifier": None, "reason": "..."},
     {"wave": "pulse on (u = 0), extremum (u = 0.39237)", "speed": 0.0, "label": "UNSTABLE", "qualifier": None, "reason": "..."},
     {"wave": "every wave that ends at (u = 0.25)", "speed": None, "label": "UNSTABLE", "qualifier": None, "reason": "..."}],
   "note": None,
   "seconds": 3.2},
  {"name": "eqn2",
   "waves": [
     {"wave": "front (u = 2) -> (u = 0)", "speed": 1.0, "label": "STABLE", "qualifier": "algebraic decay", "reason": "..."}],
   "note": None,
   "seconds": 0.5}
]
```

| Field | Content |
|---|---|
| `name` | name of the PDE |
| `waves` | the travelling waves of this PDE, one entry for each wave |
| `note` | why `waves` is empty (input error, end states are missing, no wave found, solver error); otherwise `None` |
| `seconds` | computing time of this entry |

Each wave:

| Field | Content |
|---|---|
| `wave` | kind and end states. A statement about a whole family reads `every wave that ends at (...)` |
| `speed` | wave speed `c` in `U(x - c t)`; `None` for a family |
| `label` | `STABLE`, `UNSTABLE` or `UNCERTAIN` |
| `qualifier` | a restriction that belongs to the label (below), or `None` |
| `reason` | one sentence of evidence |

The command line prints one line for each wave, `name: wave, c = speed: LABEL (qualifier)`:

```
eqn1: front (u = 0) -> (u = 1), c = -0.353553: STABLE
eqn1: pulse on (u = 0), extremum (u = 0.39237), c = 0: UNSTABLE
eqn1: every wave that ends at (u = 0.25): UNSTABLE
eqn2: front (u = 2) -> (u = 0), c = 1: STABLE (algebraic decay)
```

### 3.1 Meaning of the labels

The spectrum is that of the linearisation about the wave, in the space L2 without a weight.

| Label | Meaning |
|---|---|
| `STABLE` | The essential spectrum lies in `Re(lambda) < 0`. The eigenvalue 0 of the translation is simple. There is no other spectrum in `Re(lambda) >= 0`. |
| `UNSTABLE` | There is spectrum in `Re(lambda) > 0`: an eigenvalue (located and confirmed), or essential spectrum of an end state. |
| `UNCERTAIN` | The solver did not reach a conclusion. The printed line gives the reason. |

Qualifiers. A qualifier is part of the label and is always printed with it.

| Qualifier | Appears with | Meaning |
|---|---|---|
| `algebraic decay` | `STABLE`, conservation laws | The essential spectrum touches the origin. Nothing else lies in `Re(lambda) >= 0`. Perturbations decay algebraically, not exponentially. |
| `neutral: no spectrum with Re(lambda) > r` | `STABLE`, dispersive or undamped equations | The essential spectrum lies on the imaginary axis. There is no spectrum to the right of the small threshold `r`. |
| `default search rectangle X x Y` | `STABLE` | No estimate bounds the eigenvalues of this class. The search covered the rectangle `Re(lambda) <= X`, `abs(Im(lambda)) <= Y` only. |
| `in L2; in an exponentially weighted space: ...` | `UNSTABLE` | The instability comes from the essential spectrum only. The second label holds in a weighted space. |

Statements about families.

- `every wave that ends at (state)`: the state is unstable as a constant solution, so every wave that has it as an end state is unstable. No wave is computed.
- `every multi-pulse made of copies of the unstable pulse`: a multi-pulse inherits the unstable eigenvalue of each copy.
- `[the standing wave stands for the family of waves in the speed]`: an equation without damping has a family of waves in the speed; the standing wave is the representative.

---

## 4. How to use the solver

### 4.1 Requirements

Python with NumPy, SciPy and SymPy (tested versions: see the top of this file). No installation step: run from the folder that contains the folder `twstab`.

```
pip install numpy scipy sympy
```

### 4.2 Command line

```
python -m twstab examples/example_list.py
```

- The file must define the list `problems` (section 2).
- One line is printed for each wave as soon as its PDE is finished.
- The complete output list is written to `results/<file name>_labels.json`. Use `--out DIR` for another folder.
- `--single-pass` skips the second (audit) pass. It is faster and less safe.

The example list has 11 entries, one for each of 11 situations. It runs in about 40 seconds.

### 4.3 Python

```python
import twstab

results = twstab.solve(problems)                # problems: the list of section 2
for result in results:
    for wave in result["waves"]:
        print(result["name"], wave["wave"], wave["speed"], wave["label"], wave["qualifier"])
```

`twstab.read_list("my_list.py")` returns the list `problems` of a file. `twstab.solve(problems, on_result=f)` calls `f(result)` after each PDE.

### 4.4 Computing time

A named wave takes about 0.1 to 3 seconds. A PDE without `wave` takes about 1 to 14 seconds, because the solver searches for all waves; equations with double pulses or with a rotation symmetry are the slowest. Estimate for a list of 100 PDEs: from a few minutes to about 25 minutes. The entries are computed one after the other.

### 4.5 Tests

```
python tests/test_core.py --fast      # about 2 minutes
python tests/test_core.py             # complete set, about 7 minutes
```

Every test compares with a result that is known in closed form or from the literature.

---

## 5. Limits

- **The list of waves is not complete.** The search uses a finite set of candidates (fronts between stable constant states, pulses on them) and a finite set of guesses. A wave that is not listed can still exist. If several waves have the same end states, the search can miss some of them.
- **`UNCERTAIN` is possible.** The solver does not give a label when its checks do not agree.
- **Labels are numerical.** They are not proofs. A label holds for the given parameter values only.
- **A `STABLE` label with the qualifier `default search rectangle` holds for that rectangle only.** `UNSTABLE` labels do not depend on it.
- **Multi-pulses:** only double pulses are listed automatically.
- **Systems with a component that does not diffuse (types 2 and 9):** a wave with speed zero is not covered.
- **A named pulse of a conservation law or of a dispersive equation:** the first pulse that the guesses reach is returned. Its mirror image is not listed.
- **An entry that fails does not stop the list.** Its `waves` is empty and `note` gives the reason.

---

## 6. Content of the package

| Path | Content |
|---|---|
| `README.md` | this file |
| `twstab/` | the solver (Python package) |
| `examples/example_list.py` | example of the input format: 11 PDEs |
| `problems/`, `equations/` | 90 single-problem files used by the tests (with and without a named wave). A file of this older format holds the keys of one entry as top-level names and is also accepted: `python -m twstab problems/nagumo_front.py` |
| `tests/test_core.py` | unit tests |
| `DESIGN.md` | the design document: mathematics, numerical methods, validation (in Chinese) |
