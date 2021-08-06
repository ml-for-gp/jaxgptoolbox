# jaxgptoolbox

This is a collection of basic geometry processing functions, constructed to work with [jax](https://github.com/google/jax)'s audodifferentiation feature for applications in machine learning. We split these functions into _not differentiable_ ones in the `general` folder, and differentiable ones in the `differentiable` folder. To use these utility functions, one can simply import this package and use it as
```
import jaxgptoolbox as jgp
V,F = jgp.readOBJ('path_to_OBJ')
jgp.plotMesh(V,F)

import matplotlib.pyplot as plt
plt.show()
```

## Dependencies

This library depends on:
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [jax](https://github.com/google/jax)

If you want to use the plotting functions, they depend on
* [matplotlib](https://github.com/matplotlib/matplotlib)

Make sure to install all dependencies (for example, with [conda](https://docs.conda.io/projects/conda/en/latest/index.html))
before using the library.

## License

Consult the [LICENSE](LICENSE) file for details about the license of this project.
