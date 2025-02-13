# Streaming quadratic manifolds
## Euler simulation demo

Computes simulations of the Kelvin Helmholtz instability and stores an evolving singular value decomposition of the snapshot data. The main simulation routine which includes updating the singular value decomposition can be run using `make small_run`. This will also install the necessary virtual environment. For configuration of the virtual environment and running the simulation, see the `Makefile` and `configure.sh`.

<img width="974" alt="Screenshot 2024-09-14 at 13 00 40" src="https://github.com/user-attachments/assets/0de08ed3-ba69-41ad-a362-f5d39154c1c2">

## Related paper
```
@Article{SchwerdtnerMPBOP2024Online,
    title	= {Online learning of quadratic manifolds from streaming data for nonlinear dimensionality reduction and nonlinear model reduction},
    author	= {Paul Schwerdtner and Prakash Mohan and Aleksandra Pachalieva and Julie Bessac and Daniel O'Malley and Benjamin Peherstorfer},
    year	= {2024},
    doi		= {10.48550/arXiv.2409.02703},
    url		= {https://arxiv.org/abs/2409.02703},
    journal	= {arXiv},
    volume	= {2409.02703}
}
```
