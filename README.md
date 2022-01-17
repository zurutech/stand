# StAnD: A Dataset of Linear Static Analysis Problems

[[Abstract](https://arxiv.org/abs/2201.05356)]
[[Paper](https://arxiv.org/pdf/2201.05356.pdf)]

Static analysis of structures is a fundamental step for determining the stability of structures. Both linear and non-linear static analyses consist of the resolution of sparse linear systems obtained by the finite element method.
The development of fast and optimized solvers for sparse linear systems appearing in structural engineering requires data to compare existing approaches, tune algorithms or to evaluate new ideas.
We introduce the Static Analysis Dataset (StAnD) containing 303.000 static analysis problems obtained applying realistic loads to simulated frame structures. 
Along with the dataset, we publish a detailed benchmark comparison of the running time of existing solvers both on CPU and GPU. We release the code used to generate the dataset and benchmark existing solvers on Github.
To the best of our knowledge, this is the largest dataset for static analysis problems and it is the first public dataset of sparse linear systems (containing both the matrix and a realistic constant term).

---

## How to download the dataset

StAnD is publicly hosted on Google Cloud Storage in 3 separate .zip files:
one archive for every part of the dataset i.e. small, medium and large problems.
Every problem is saved in a separate `.npz` file.
Every `.zip` file contains 2 folders:
- `stand_<size>_train`: training set composed of 100.000 `<size>` problems;
- `stand_<size>_test`: test set composed of 1.000 `<size>` problems;
where `<size>` is `small`, `medium` or `large`.

The dataset can be downloaded through `gsutil`
```
gsutil cp gs://zurutech-public-datasets/stand/stand_small.zip .
gsutil cp gs://zurutech-public-datasets/stand/stand_medium.zip .
gsutil cp gs://zurutech-public-datasets/stand/stand_large.zip .
```
or using the public URLs:
- [StAnD (Small problems)](https://storage.googleapis.com/zurutech-public-datasets/stand/stand_small.zip);
- [StAnD (Medium problems)](https://storage.googleapis.com/zurutech-public-datasets/stand/stand_medium.zip);
- [StAnD (Large problems)](https://storage.googleapis.com/zurutech-public-datasets/stand/stand_large.zip).

The 3 parts of StAnD are also available on Kaggle:
- https://www.kaggle.com/zurutech/stand-small-problems
- https://www.kaggle.com/zurutech/stand-medium-problems
- https://www.kaggle.com/zurutech/stand-large-problems

## How to load a .npz file

The code below shows how to read correctly a `.npz` file whose path is stored in variable `path` to restore the linear system.

```python
# Read system file and build sparse stiffness matrix
with np.load(path) as system:
    u = system["x"]
    f = system["b"]
    K = scipy.sparse.coo_matrix(
        (system["A_values"], list(system["A_indices"])),
        shape=(u.size, u.size)
    )

# Check K * u = f
if(np.allclose(K.dot(u), f)):
    print("Everything is correct!")
else:
    print("There is something wrong!")
```

## Replicating benchmark results

Replicating the results is essential. For this reason, we created a docker image ready to use.

**NOTE**: the benchmarks require a NVIDIA GPU, therefore you need to configure the docker NVIDIA runtime.

1. Pull the docker image

```
sudo docker pull europe-west3-docker.pkg.dev/machine-learning-199407/structures/stand:latest
```

2. Download a dataset in the `datasets` subdirectory of the repository.

```bash
mkdir datasets
cd datasets
wget https://storage.googleapis.com/zurutech-public-datasets/stand/stand_small.zip
unzip stand_small.zip
cd ..
```

3. Run an interactive shell mounting the repository (where this README is) in the `/home/aur/env` folder

```bash
sudo docker run -it \
           -v $(pwd):/home/aur/env \
           --runtime=nvidia \
           --device /dev/nvidia-caps \
           --device /dev/nvidia0 \
           --device /dev/nvidiactl \
           --device /dev/nvidia-modeset \
           --device /dev/nvidia-uvm \
           --device /dev/nvidia-uvm-tools \
           europe-west3-docker.pkg.dev/machine-learning-199407/structures/stand:latest
```

4. You're now inside the container. Please ensure the NVIDIA runtime is working correctly by calling `nvidia-smi`.

5. If `nvidia-smi` shows your GPU info, then you are ready for building and executing the benchmarks. In the example below we run the benchmark on the test set of StAnD small problems.

```bash
cd /home/aur/env/solvers

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARK=ON ..
make -j$(nproc)

# Run tests to check if everything is fine
CTEST_OUTPUT_ON_FAILURE=1 make test

# If there are no errors (otherwise check your docker configuration)
# you can run the benchmarks
./benchmark/solvers_benchmark --dataset=../../datasets/stand_small_test
```

## Troubleshooting

If you see this error while running the benchmarks

```
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
```

you might want to disable the CPU frequency scaling while running the benchmark (on the **host** not inside the container):

```bash
sudo cpupower frequency-set --governor performance
./benchmark/solvers_benchmark --dataset=<dataset_directory>
sudo cpupower frequency-set --governor powersave
```

---

## Citation

If you find the dataset or the code useful for your research, please cite:

```
@article{grementieri2022stand,
    title={StAnD: A Dataset of Linear Static Analysis Problems}, 
    author={Grementieri, Luca and Finelli, Francesco},
    journal={arXiv preprint arXiv:2201.05356},
    year={2022}
}
```
