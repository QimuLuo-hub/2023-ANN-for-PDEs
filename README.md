# Artificial Neural Networks for PDEs (2023–2024)

Reports, reading notes and code from a 2023–2024 study of neural-network solvers for the **2D Poisson equation** $\Delta u = f$ on $(0,1)^2$. It began as a study group on deep-learning fundamentals and scientific computing at XJTLU.

The project had two stages:

1. **GAN-style two-network solver (Nov 2023).** Following *Deep learning for PDEs*, the solution is written as
   $$u(x,y) = A(x,y; w_1) + B(x,y)\,N(x,y; w_2), \qquad B = x(1-x)\,y(1-y),$$
   where a boundary network $A$ fits the Dirichlet data and a PDE network $N$ fits the interior. In each iteration the boundary network is updated $k$ times before the PDE network is updated once.
2. **Physics-informed neural networks (Dec 2023 – Jun 2024).** A single network is trained on the PDE residual plus weighted boundary losses. It is tested with **Dirichlet**, **Neumann** and **Robin** boundary conditions, and the effect of iterations, training-set size, architecture and boundary-loss weights is compared.

## Contents

| | |
|---|---|
| [`Report&Notes/Mid-report_ANN_for_PDE_First_Trial.pdf`](<Report&Notes/Mid-report_ANN_for_PDE_First_Trial.pdf>) | Mid-term report: GAN-style boundary/PDE networks for the Poisson equation (15 pp.) |
| [`Report&Notes/final_report_Solve_Poisson_with_PINN.pdf`](<Report&Notes/final_report_Solve_Poisson_with_PINN.pdf>) | **Final report**: solving the Poisson equation with PINNs under three boundary conditions (19 pp.) |
| [`Report&Notes/Reading_Notes.pdf`](<Report&Notes/Reading_Notes.pdf>) | Reading notes (in Chinese) on machine-learning fundamentals: linear models, bias–variance, evaluation metrics, learning theory (83 pp.) |
| [`Python codes/`](<Python codes/>) | PyTorch PINN scripts used in the final report |

### Code

| Script | Description |
|---|---|
| `PINN_Dirichlet(seperate).py` | Dirichlet problem; boundary sides added as separate loss terms |
| `PINN_Neumann(seperate).py` | Dirichlet on $y=0$ and zero Neumann on the other sides, with weighted losses |
| `PINN_Robin(seperate).py` | Adds a Robin condition $\partial_n u + u = 0$ on $y = 1$ |
| `Comparison_*(integrate).py` | Parameter sweeps (epochs, $N_b$/$N_f$, width/depth, weights) that save error/runtime tables |

```bash
pip install torch numpy matplotlib pandas
cd "Python codes"
python "PINN_Robin(seperate).py"
```

## Main observations (from the final report)

- The **boundary-loss weights** had the largest effect on accuracy in the Neumann and Robin cases. This agrees with the NTK analysis of Wang et al. (2022).
- Re-sampling the collocation points at every iteration gave better coverage than a fixed training set.
- Lowering the learning rate reduced error fluctuations, but more iterations were then needed.
- **Limitations:** the error still fluctuates noticeably during training, and the effects of the loss function, activation and optimiser were not studied.

A cleaned-up copy of the code, with some later experiments, is in [PINN-PDE](https://github.com/QimuLuo-hub/PINN-PDE).

## References

- M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *J. Comput. Phys.* 378 (2019) 686–707.
- S. Wang, X. Yu, P. Perdikaris. When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449 (2022) 110768.
- T. G. Grossmann, U. J. Komorowska, J. Latz, C.-B. Schönlieb. Can physics-informed neural networks beat the finite element method? arXiv:2302.04107 (2023).
