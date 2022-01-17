# Copyright 2022 Zuru Tech HK Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# StAnD Dockerfile
#
# Create a reproducible environment for running the benchmarks.

# You won't be able to `docker build` this Dockerfile outside our Local Area Network
# because we rely upon a Local ArchLinux User Repository that contains all the packages already patched
# and pre-built.

# You can, anyway, use this Dockerfile to see what packages have been installed
# and use the publicly available docker image (you are encouraged to pull) for replicating the benchmarks.

FROM archlinux
LABEL maintainer="Paolo Galeone <paolo@zuru.tech>"

# Add custom archlinux repository (local AUR).
RUN echo -e "[laur]\nSigLevel = Never\nServer = http://gitserver.zurutechitaly.local/ml/laur/-/raw/main/x86_64" >> /etc/pacman.conf

# Init the pacman stuff and upgrade the package and keys databases
RUN pacman -Syy pacman reflector haveged archlinux-keyring --noconfirm && \
    haveged -w 1024 -v 1 && \
    pacman-key --init && \
    pacman-key --populate archlinux && \
    pacman-db-upgrade

# use reflector to get the fastest repo and update system
RUN reflector --latest 5 --protocol http,https --sort rate --save /etc/pacman.d/mirrorlist && pacman -Syyu --noconfirm

# Install basic system libaries and basic system tools
RUN yes | pacman -S systemd-libs && \
    pacman -Syu rsync base-devel glibc libidn wget git libevent unzip --noconfirm && \
    yes | pacman -Scc

# Create a "aur" user, add it to the sudoers without pass
RUN useradd -m -s /bin/bash aur && \
    echo "aur ALL = NOPASSWD: /usr/bin/pacman" >> /etc/sudoers && \
    echo "aur ALL = NOPASSWD: /usr/bin/make" >> /etc/sudoers

# Update the system makeflags to use (more or less) the number of available cores
RUN echo 'MAKEFLAGS="-j$(nproc)"' >> /etc/makepkg.conf

# Install yay (aur helper) from laur
RUN sudo -u aur bash -c 'sudo pacman -S laur/yay --noconfirm && yes | sudo pacman -Scc'

# Install the compiler
RUN sudo -u aur bash -c 'yay -S clang --noconfirm && yes | yay -Scc'

## C++ libraries
RUN sudo -u aur bash -c 'yay -S eigen cuda fmt parmetis viennacl openmpi benchmark gtest --noconfirm && yes | yay -Scc'
RUN sudo -u aur bash -c 'yay -S laur/suitesparse laur/superlu_dist-cuda laur/petsc-cuda --noconfirm && yes | yay -Scc'
RUN sudo -u aur bash -c 'yay -S hdf5 intel-mkl --noconfirm && yes | yay -Scc'

# Install gcloud sdk from aur (aur patches the sdk to use python3)
RUN sudo -u aur bash -c 'yay -Sy google-cloud-sdk --noconfirm && yes | yay -Scc'

# Cleanup installed cuda libraries (will be provided by nvidia-docker) samples and local cache
RUN rm -vf /usr/bin/nvidia* && rm -vf /usr/lib/libnvidia* && rm -vf /usr/lib/libcuda* && \
    rm -rvf /opt/cuda/doc/ && rm -rvf /opt/cuda/*nsight* && rm -rvf /opt/cuda/*nvvp* && \
    rm -rvf /opt/cuda/samples/

# Set the container user as aur (not using root) and create an env folder that can be used from the outside (if needed)
USER aur
RUN cd /home/aur && mkdir env && cd env

VOLUME /home/aur/env
ENTRYPOINT /usr/bin/bash
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11"
# Set the correct cuda env vars
# Linking time is different from compile time (LD_LIBRARY_PATH != from LIBRARY_PATH)
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64
ENV LIBRARY_PATH=/opt/cuda/lib64/stubs:/opt/cuda/targets/x86_64-linux/lib/stubs/
ENV CUDA_HOME=/opt/cuda
ENV CUDA_PATH=/opt/cuda
ENV PATH=$PATH:/opt/cuda/bin:/opt/cuda/nsight_compute:/opt/cuda/nsight_systems/bin
