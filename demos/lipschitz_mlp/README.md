# Learning Smooth Neural Functions via Lipschitz Regularization

This is the demo code for the Lipschitz MLP:

**Learning Smooth Neural Functions via Lipschitz Regularization**
_Hsueh-Ti Derek Liu, Francis Williams, Alec Jacobson, Sanja Fidler, Or Litany_
SIGGRAPH (North America), 2022 
[[Project Page](https://nv-tlabs.github.io/lip-mlp/)] [[Preprint](https://www.dgp.toronto.edu/~hsuehtil/pdf/lipmlp.pdf)]

### Dependencies
Our method depends on [JAX](https://github.com/google/jax) and some common python dependencies (e.g., numpy, tqdm, matplotlib, etc.). Some functions in the script, such as generating analytical signed distance functions, depend on other parts in the repository -- [jaxgptoolbox](https://github.com/ml-for-gp/jaxgptoolbox).

### Repository Structure
- `main_lipmlp.py` is the main training script. This is a self-contained script to train a Lipschitz MLP to interpolate 2D signed distance functions of a star and a circle. To train the model from scratch, one can simply run
```python
python main_lipmlp.py
```
After training (~15 min on a CPU), you should see the interpolation results in `lipschitz_mlp_interpolation.mp4` and the model parameters in `lipschitz_mlp_params.pkl`.
- `model.py` contains the Lipschitz MLP model. One can simply use it as
```python
model = lipmlp(hyper_params) # build the model
params = model.initialize_weights() # initialize weights
y = model.forward(params, latent_code, x) # forward pass
```

