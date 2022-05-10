# jaxgptoolbox

This is a collection of basic geometry processing functions, constructed to work with [jax](https://github.com/google/jax)'s audodifferentiation feature for applications in machine learning. We split these functions into _not differentiable_ ones in the `general` folder, and differentiable ones in the `differentiable` folder. To use these utility functions, one can simply import this package and use it as
```
import jaxgptoolbox as jgp
import polyscope as ps
V,F = jgp.read_mesh('path_to_OBJ')
ps.init()
ps.register_surface_mesh('my_mesh',V,F)
ps.show()
```

### Dependencies

This library depends on [jax](https://github.com/google/jax) and some common python libraries [numpy](https://github.com/numpy/numpy)  [scipy](https://github.com/scipy/scipy). Our `demos` rely on [matplotlib](https://github.com/matplotlib/matplotlib) and [polyscope](https://polyscope.run/py/) for visualization. Some functions in the `external` folder depend on [libigl](https://libigl.github.io/libigl-python-bindings/). Please make sure to install all dependencies (for example, with [conda](https://docs.conda.io/projects/conda/en/latest/index.html)) before using the library.

### Contacts & Warnings

The toolbox grew out of [Oded Stein](https://odedstein.com)'s and [Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/)'s private research codebase during their PhD studies. Some of these functions are not fully tested nor optimized, please use them with caution. If you're interested in contributing or noticing any issues, please contact us (ostein@mit.edu, hsuehtil@cs.toronto.edu) or submit a pull request.

### License

Consult the [LICENSE](LICENSE) file for details about the license of this project.
