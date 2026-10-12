from setuptools import setup
#from Cython import cythonize
from Cython.Build.Dependencies import cythonize

setup(
    name="SwarmFish",
    packages=["swarmfish"],
    version="0.1.0",
    install_requires=[
        "numpy",
    ],
    ext_modules=cythonize(["swarmfish/swarm_control.py","swarmfish/obstacles.py"]),
)
