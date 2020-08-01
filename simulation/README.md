### This tutorial will help to compile your neuron model in NEST.
Two of the famous neuron simulators is NEST and BRAIN. BRAIN is user friendly, howerver it cannot suppport large-scale network. In other words, BRAIN is TOO MORE SLOWER than NEST. We choose NEST as our default simulator. 

Even using NEST, you still need to be careful with the scale of your network. One thread in NEST only support limit neurons, so you'd better set "local_num_threads" with the statement like ``` nest.SetKernelStatus({"local_num_threads": threads}) ```. Additionally, the order of connections returned by ```nest.GetConnections()``` cann't be promised when the number of threads more than one.

Unfortunately, the models in NEST cannot satisfy our simulation. So, we need compile the source code of NEST by adding our model. This means you need to check your model too. Fllowing is the progress of the compilation. Good Luck!

```
1. Download nest-simulator-2.18.0.tar.gz and unzip the file
2. Add your model.h and model.cpp in "models/"
3. Modify some config files
- models/modelsmodule.cpp：add your model.h and register your model
- models/CMakeLists.txt：add your model.h and model.cpp
- nestkernel/nest_names.cpp: add all variables you defined
- nestlernel/nest_names.h：add all variables you defined
4. Source conda env(python3.6)
5. Open terminal, and enter
>cmake -DCMAKE_INSTALL_PREFIX:PATH=install_path src_path
>Make
>make install 
info: install_path is a new diretions you used to build nest, and src_path is the unzip direction
6. Replace the file in your conda env with the compiled files, which maybe in "opt/"
```

If you still confuse about the progess, please review the files in "example\" for more help. The example neuron model is defined in "example\iaf_delta_noleak.h" and "example\iaf_delta_noleak.cpp".

If you don't want to compile the NEST, you can also using my compiled files which can be download from [google drive](https://drive.google.com/file/d/1HUlGNSmagTSvkIlJ-4IMdUM2C5cdshlD/view?usp=sharing) or [baidudisk with code "2snn"](https://pan.baidu.com/s/1AN-FSSYyWcesoOqYQ3VLFA ). Howerer, I cannot promise it would work for you due to the diffrence between our systems or other factors.
