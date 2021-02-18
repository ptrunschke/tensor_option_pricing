#!/bin/bash
ENVNAME="${1}"
set -e

if [ -z "${ENVNAME}" ];
then
    echo "Usage: bash install.sh <env_name>"
    exit 0
fi

read -p "Create new conda environment '${ENVNAME}' (y/n)? " answer
case ${answer:0:1} in
    y|Y )
    ;;
    * )
        exit
    ;;
esac

# conda create -n ${ENVNAME} -c conda-forge 'python=3.8' python_abi gxx_linux-64 make 'pip>=18.1' numpy openblas suitesparse lapack liblapacke boost-cpp libgomp scipy matplotlib rich
eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
NUMPY=${CONDA_PREFIX}/lib/python3.8/site-packages/numpy
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++

git clone --recurse-submodules https://github.com/libxerus/xerus.git --branch SALSA
cd xerus

cat <<EOF >config.mk
CXX = ${CXX}
COMPATIBILITY = -std=c++17
COMPILE_THREADS = 8                       # Number of threads to use during link time optimization.
HIGH_OPTIMIZATION = TRUE                  # Activates -O3 -march=native and some others
OTHER += -fopenmp

PYTHON3_CONFIG = \`python3-config --cflags --ldflags\`

LOGGING += -D XERUS_LOG_INFO              # Information that is not linked to any unexpected behaviour but might nevertheless be of interest.
LOGGING += -D XERUS_LOGFILE               # Use 'error.log' file instead of cerr
LOGGING += -D XERUS_LOG_ABSOLUTE_TIME     # Print absolute times instead of relative to program time
XERUS_NO_FANCY_CALLSTACK = TRUE           # Show simple callstacks only

BLAS_LIBRARIES = -lopenblas -lgfortran    # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack     # Standard Lapack + Lapacke libraries
SUITESPARSE = -lcholmod -lspqr
BOOST_LIBS = -lboost_filesystem

OTHER += -I${CONDA_PREFIX}/include -I${NUMPY}/core/include/
OTHER += -L${CONDA_PREFIX}/lib
EOF

ln -s ${CONDA_PREFIX}/include/ ${CONDA_PREFIX}/include/suitesparse
make python
python -m pip install . --no-deps -vv

cd ..
rm -rf xerus
