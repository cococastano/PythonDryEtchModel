# PythonDryEtchingTool
Code was developed for SNF at Stanford University in conjunction with the E241 course. These scripts are meant to predict etch profiles from dry silicon etching based on know etch rates. Two versions are provided:
..* Version 0: Simple implementation which utilized polygon and path objects. This version grows down and out layer by layer but neglects to evolve layers constructed at previous timepoints, i.e., a Bosch step at the beginning will grow down but after this layer is formed, subsuquent etching steps will not change its shape/form. This is useful for seeing the result of horizontally moving etch fronts and how they interact with each other. <br/>

![picture alt](./Figures/verion0_example.gif)

..* Version 1 (under development): More elegant implementation that evolves a surface object with easily recipe steps. This version is tailored for precribing custom bsoch, isotropic, and taperd (combined bosch and isotropic) etching steps. The surface is evolved by using the computed normals  of the surface and stepping points back along the normals by some user defined vertical and horizontal etch rates. Because this version used vtk based rendering tools, this model outputs nice interactive renders that can be saved as vtk files for later usage. Below is an example screenshot from a 90 um bosch etch (code under development). <br/>

![picture alt](./Figures/verion1_example.png)

## Environment
Python 3.6 was used for developing these scripts. My personal preference is using Spyder IDE in the Anaconda environment. The solvers rely heavily on some imported python packages, so you environment should have the following installed: openCV (cv2), shapely, and pyvista. I encourage you to familiarize yourself with documentation and install instructions for each package, but in particular pyvista has a number of dependencies, most notably vtk, and some others, for full functionality, include: imageio, appdirs, and meshio.

All these packages can be directly installed from the command line (i.e., Anaconda command line) with pip. For example:
```
pip install vtk
pip install opencv-python
```

## Process Flow
Most input expected from the user is specified in the top sections of the code. Most notably the path to a etch mask is required. This is a binary .png file (example shown below). <br/>

![picture alt](./ExampleMasks/fillet_sq_example_mask.png)
 
Here is will describe the Verion 1 code as it is meant to replace Version 0. After the mask is provided, tune the desired etch rates in the vertical and horizontal directions, the curved profile of the isotropic etch will be interpolated. Other parameters can be set, like time step (t_step) or resoltuion of mesh (set_res) that affect solution time. 

## Future of the Code
I am not a coder, so my code is messy. I would be happy for a savy coder to come along and clean up this work and maybe create more elegant classes. For SNF users, eventually I want there to be enough data that this tool can take in recipe settings (i.e. gas composition and bias voltage) and interpolate horizontal and vertical etch rates. 
