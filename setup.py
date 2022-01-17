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

#!/usr/bin/env python3

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "matplotlib",
    "networkx",
    "numpy",
    "openseespy>=3.4",
    "opsvis>=0.96.2",
    "pyyaml",
    "scipy",
    "tqdm",
]


setup(
    author="Zuru Tech Machine Learning Team",
    author_email="ml@zuru.tech",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Engineers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Static Analysis Dataset",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords="static analysis dataset finite elements",
    name="StAnD",
    packages=find_packages(include=["stand", "stand.*"]),
    url="https://github.com/zurutech/stand",
    version="1.0",
    zip_safe=False,
)
