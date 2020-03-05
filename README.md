# rlly - A C++ library for reinforcement learning environments (under development)

Our goal is to provide a C++ alternative to [OpenAI gym](https://gym.openai.com/), starting with the simplest environments.

## Requirements

* C++ 17

* To run examples with jupyter notebook, we need:
    * Jupyter notebook `conda install -c anaconda jupyter`
    * [xeus-cling](https://xeus-cling.readthedocs.io/en/latest/)  `conda install -c conda-forge xeus-cling`

* For `rlly_rendering`, we need [freeglut](http://freeglut.sourceforge.net/) and OpenGL. To install freeglut, run
    ```console
    $ sudo apt-get install freeglut3-dev
    ```

## How to use the library

All you have to do is to copy the file `rlly.hpp` and use it in your project.

If you modify the source code, you need to regenerate the file by running

```console
$ bash generate_header/run.sh
```

## Examples

See `examples/rlly_introduction.ipynb`.


## Documentation

To view the documentation, run

```
doxygen Doxyfile
```

and open the file `docs/html/index.html`.


## Testing

### Creating a new test

* Create a file `test/my_test.cpp` using [Catch2](https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md).

* In the file `test/CMakeLists.txt`, include `my_test.cpp` in the list of files in `add_executable()`.

* Run

```
$ bash scripts/run_tests.sh
```

