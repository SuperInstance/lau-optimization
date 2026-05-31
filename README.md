# lau-optimization

General optimization algorithms for finding minima of objective functions.

## Features

- **Line Search**: Golden section, backtracking Armijo, Wolfe conditions
- **Gradient Descent**: Vanilla, stochastic (SGD), mini-batch, momentum, Nesterov
- **Conjugate Gradient**: Fletcher-Reeves, Polak-Ribiere
- **Quasi-Newton**: BFGS, L-BFGS
- **Nelder-Mead**: Derivative-free simplex method
- **Simulated Annealing**: Probabilistic global optimization with configurable temperature schedules
- **Particle Swarm**: PSO metaheuristic for continuous optimization
- **Multi-Objective**: Pareto fronts, NSGA-II with crowding distance
- **Constraint Handling**: Quadratic penalty method, log-barrier method
- **Hyperparameter Tuning**: Optimize agent configurations using any available method

## Usage

```rust
use lau_optimization::gradient_descent::{gradient_descent, GradientDescentConfig};
use lau_optimization::test_functions::{sphere, sphere_grad};
use nalgebra::DVector;

let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0]);
let config = GradientDescentConfig::default();
let result = gradient_descent(&sphere, &sphere_grad, &x0, &config);
println!("Minimum at: {:?}, value: {}", result.x, result.f_x);
```

## Test Functions

Built-in benchmark functions: Sphere, Rosenbrock, Rastrigin, Booth, Ackley.

## License

MIT
