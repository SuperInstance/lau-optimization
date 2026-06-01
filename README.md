# lau-optimization

A pure-Rust numerical optimization library covering classical gradient-based methods, derivative-free heuristics, multi-objective optimization, and constraint handling.

**87 tests** · `nalgebra` + `serde` + `rand` · MIT license

---

## What This Does

`lau-optimization` gives you a toolbox of optimization algorithms you can call with a few lines of Rust. Every optimizer operates on the same interface — a user-supplied objective function `f: &DVector<f64> -> f64` (and optionally its gradient) — so you can swap algorithms without rewriting your problem.

The crate is **no-std-friendly in spirit** (no I/O, no filesystem, no networking) and depends only on `nalgebra` for linear algebra, `serde` for serialization, and `rand` for stochastic methods.

---

## Key Idea

Most numerical optimization crates are either *specialised* (one algorithm, feature-rich) or *thin wrappers around C/Fortran*. This crate is a **self-contained, idiomatic-Rust collection** of the most widely-taught algorithms from a graduate optimization course:

| Category | Algorithms |
|---|---|
| **Line search** | Backtracking (Armijo), Wolfe conditions, golden-section |
| **First-order** | Gradient descent (constant / diminishing step, momentum / Nesterov) |
| **Conjugate gradient** | Fletcher–Reeves, Polak–Ribière |
| **Quasi-Newton** | BFGS, L-BFGS (limited-memory) |
| **Derivative-free** | Nelder–Mead simplex, simulated annealing, particle swarm optimization |
| **Multi-objective** | NSGA-II (Non-dominated Sorting Genetic Algorithm II) |
| **Constraints** | Penalty methods, barrier methods |
| **Hyperparameter tuning** | Grid search, random search |
| **Test functions** | Rosenbrock, Rastrigin, Ackley, Sphere, Beale, Booth, Himmelblau, etc. |

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-optimization = "0.1"
nalgebra = "0.33"
```

Or via CLI:

```sh
cargo add lau-optimization
```

---

## Quick Start

### Minimise Rosenbrock with L-BFGS

```rust
use lau_optimization::test_functions::rosenbrock;
use lau_optimization::test_functions::rosenbrock_gradient;
use lau_optimization::quasi_newton::lbfgs;
use nalgebra::DVector;

fn main() {
    let start = DVector::from_vec(vec![-1.0, 2.0]);
    let (x_min, f_min) = lbfgs(
        &start,                           // initial guess
        |x| rosenbrock(x),                // objective f(x)
        |x| rosenbrock_gradient(x),       // gradient ∇f(x)
        10,                               // L-BFGS memory (m)
        1000,                             // max iterations
        1e-10,                            // gradient tolerance
    );
    println!("x* ≈ {:.6}, f* ≈ {:.10}", x_min, f_min);
}
```

### Particle Swarm on Rastrigin (derivative-free)

```rust
use lau_optimization::particle_swarm::pso;
use lau_optimization::test_functions::rastrigin;
use nalgebra::DVector;

let (best, f_best) = pso(
    30,                                 // swarm size
    2,                                  // dimension
    |x| rastrigin(x),                   // objective
    vec![(-5.12, 5.12); 2],            // bounds per dimension
    500,                                // max iterations
    0.9, 0.5, 0.3,                      // inertia, cognitive, social weights
);
```

---

## API Reference

### Line Search (`line_search`)

| Function | Signature | Description |
|---|---|---|
| `backtracking` | `(f, grad_f, x, direction, alpha, rho, c) → alpha` | Armijo backtracking line search |
| `wolfe` | `(f, grad_f, x, direction, c1, c2) → alpha` | Strong Wolfe line search |
| `golden_section` | `(f, a, b, tol) → (x, f(x))` | Derivative-free 1-D minimisation |

### Gradient Descent (`gradient_descent`)

| Function | Description |
|---|---|
| `gradient_descent` | Constant step-size gradient descent |
| `gradient_descent_diminishing` | Diminishing step α_k = 1/k |
| `gradient_descent_momentum` | Heavy-ball / momentum method |
| `gradient_descent_nesterov` | Nesterov accelerated gradient |

### Conjugate Gradient (`conjugate_gradient`)

| Function | Description |
|---|---|
| `conjugate_gradient_fr` | Fletcher–Reeves CG with line search |
| `conjugate_gradient_pr` | Polak–Ribière CG with line search |

### Quasi-Newton (`quasi_newton`)

| Function | Description |
|---|---|
| `bfgs` | Full BFGS with inverse Hessian update |
| `lbfgs` | Limited-memory BFGS (specify memory *m*) |

### Nelder–Mead (`nelder_mead`)

| Function | Description |
|---|---|
| `nelder_mead` | Downhill simplex (no gradient needed) |

### Simulated Annealing (`simulated_annealing`)

| Function | Description |
|---|---|
| `simulated_annealing` | Metropolis–Hastings with exponential cooling schedule |

### Particle Swarm (`particle_swarm`)

| Function | Description |
|---|---|
| `pso` | Classic PSO with inertia-weight velocity update |

### Multi-Objective (`multi_objective`)

| Function | Description |
|---|---|
| `nsga2` | NSGA-II: fast non-dominated sort + crowding distance |

Returns a `Vec<Vec<DVector<f64>>>` — Pareto fronts, sorted by rank.

### Constraints (`constraints`)

| Function | Description |
|---|---|
| `penalty_method` | Quadratic penalty: add μ·max(0, g(x))² to objective |
| `barrier_method` | Log-barrier for inequality constraints |

### Hyperparameter Tuning (`hyperparameter`)

| Function | Description |
|---|---|
| `grid_search` | Exhaustive search over a discrete grid |
| `random_search` | Random sampling over a bounded space |

### Test Functions (`test_functions`)

Objective functions with known global minima, useful for benchmarking:

| Function | Minimum | Domain |
|---|---|---|
| `sphere` | 0 at origin | ℝⁿ |
| `rosenbrock` | 0 at (1,…,1) | ℝⁿ |
| `rastrigin` | 0 at origin | [-5.12, 5.12]ⁿ |
| `ackley` | 0 at origin | ℝⁿ |
| `beale` | 0 at (3, 0.5) | ℝ² |
| `booth` | 0 at (1, 3) | ℝ² |
| `himmelblau` | 0 (four minima) | ℝ² |
| `goldstein_price` | 3 at (0, -1) | ℝ² |
| `matyas` | 0 at origin | ℝ² |
| `levi` | 0 at (1, 1) | ℝ² |

Each function has a companion `_gradient` version (e.g. `rosenbrock_gradient`).

---

## How It Works

### Architecture

Every optimizer is a **pure function** — it takes the problem definition (objective, gradient, bounds, etc.) and hyperparameters, then returns `(argmin, f_min)`. There is no internal mutable state or builder pattern. This makes the API predictable and easy to test.

### Line Search → Gradient Descent → CG → Quasi-Newton

The crate mirrors the classic pedagogical progression:

1. **Line search** finds the step size α along a descent direction.
2. **Gradient descent** uses ∇f as the direction — simple but slow (O(1/k)).
3. **Conjugate gradient** improves directions using previous gradient info — O(n) storage.
4. **BFGS / L-BFGS** builds an approximate inverse Hessian — superlinear convergence, L-BFGS uses O(mn) memory.

### Derivative-Free Methods

When gradients are unavailable:

- **Nelder–Mead** reflects/contracts a simplex through function evaluations.
- **Simulated annealing** randomly perturbs the solution, accepting worse moves with probability e^{-Δf/T} where T decays over time.
- **Particle swarm** maintains a population of "particles" with velocities influenced by personal and global best positions.

### Multi-Objective: NSGA-II

NSGA-II sorts the population into Pareto fronts (fast non-dominated sort), then selects by crowding distance to maintain diversity. The result is a set of trade-off solutions rather than a single optimum.

---

## The Math

### Wolfe Conditions

A step size α satisfies the **strong Wolfe conditions** if:

- f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp (sufficient decrease)
- |∇f(x + αp)ᵀp| ≤ c₂|∇f(x)ᵀp| (curvature condition)

with 0 < c₁ < c₂ < 1.

### BFGS Update

The inverse Hessian approximation H_k is updated via:

**H_{k+1} = (I - ρₖ sₖyₖᵀ) Hₖ (I - ρₖ yₖsₖᵀ) + ρₖ sₖsₖᵀ**

where sₖ = x_{k+1} − xₖ, yₖ = ∇f_{k+1} − ∇fₖ, ρₖ = 1/(yₖᵀsₖ).

### NSGA-II Non-Dominated Sort

Solutions are ranked by Pareto dominance: solution **a** dominates **b** if a is ≤ b in all objectives and strictly < in at least one. Rank 1 = Pareto front; rank 2 = front after removing rank 1; etc.

### Simulated Annealing Acceptance

Accept a worse state with probability **P = exp(−Δf / T)**, where T is the temperature that decreases geometrically: T_{k+1} = α · Tₖ (0 < α < 1).

---

## License

MIT
