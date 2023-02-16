# ThreeBodyProblemExamples

[![Build Status](https://github.com/jared711/ThreeBodyProblemExamples.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jared711/ThreeBodyProblemExamples.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jared711/ThreeBodyProblemExamples.jl/main)

# ThreeBodyProblemExamples
This repository contains helpful examples, both in Julia scripts and Jupyter notebooks for working with ThreeBodyProblem.jl.
If you are brand new to the package, go through the examples in order to get a feel for how everything works.
There are some elements of the package that are not included in the examples, but can be found in the documentation.
Each example has a .jl file and a .ipynb file. The .jl file can be run line by line in your favorite IDE while the .ipynb file is a Jupyter notebook.

## Jupyter Notebooks
To run Jupyter notebooks on your local machine, you need to have the IJulia package downloaded.
```julia
Pkg.add("IJulia")
using IJulia
notebook(dir=".",detached=true)
```

You can also run the Jupyter notebooks on your browser through binder. Just click the binder icon
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jared711/ThreeBodyProblemExamples.jl/main) and wait for a JupyterLab window to appear. It may take a few minutes to load. Eventually this window should appear.
![image](https://user-images.githubusercontent.com/25643720/216104189-4d60e01b-dc72-4946-b72f-0d774bd78187.png)
Pick the Jupyter notebook (filetype .ipynb) you want to run. Pressing CTRL+ENTER will run a block of code, while SHIFT+ENTER will run a block and move to the next one.

Check out the documentation or the source code itself for more tips.

