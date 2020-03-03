# rlly - A C++ library for reinforcement learning environments (under development)

## Requirements

* C++ 17

* To run examples with jupyter notebook, we need:
    * Jupyter notebook `conda install -c anaconda jupyter`
    * [xeus-cling](https://xeus-cling.readthedocs.io/en/latest/)  `conda install -c conda-forge xeus-cling`

## How to use the library

All you have to do is to copy the file `rlly.hpp` and use it in your project.

If you modify the source code, you need to regenerate the file by running

```console
$ bash generate_header/run.sh
```

## Examples

See `examples/rlly_introduction.ipynb`.
