#!/bin/bash
ENVNAME="${1}"
BRANCHNAME="${2:-conda}"
set -e

if [ -z "${ENVNAME}" ]; then
    echo "Usage: bash install.sh <environment> [<branch>]"
    echo "       If <branch> is not provided the 'conda' branch will be used."
    exit 0
fi

if ! command -v conda &> /dev/null; then
    echo "Please install conda before running this script."
    exit 0
fi

eval "$(conda shell.bash hook)"

REQUIREMENTS="'python=3.8' python_abi gxx_linux-64 make 'pip>=18.1' numpy openblas suitesparse lapack liblapacke boost-cpp libgomp scipy matplotlib rich"
if test -n "$(conda env list | grep ${ENVNAME})"; then
    echo "Conda environment '${ENVNAME}' already exists."
    read -p "Use this environment? (y/n) " answer
    case ${answer:0:1} in
        y|Y )
            conda activate ${ENVNAME}
            conda install -y -c conda-forge ${REQUIREMENTS}
        ;;
        * )
            exit
        ;;
    esac
else
    read -p "Create new conda environment '${ENVNAME}'? (y/n) " answer
    case ${answer:0:1} in
        y|Y )
            conda create -y -n ${ENVNAME} -c conda-forge ${REQUIREMENTS}
            conda activate ${ENVNAME}
        ;;
        * )
            exit
        ;;
    esac
fi

echo "Installing branch 'xerus/${BRANCHNAME}' into environment '${ENVNAME}'."

CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
NUMPY=$(python -c 'import numpy as np; print("/".join(np.__file__.split("/")[:-1]))')
XERUS=$(python -c 'import numpy as np; print("/".join(np.__file__.split("/")[:-2]))')/xerus.so

cd /tmp
TEMPDIR=$(mktemp -d)
cd ${TEMPDIR}
git clone --recurse-submodules https://github.com/libxerus/xerus.git --branch ${BRANCHNAME}
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

if [ ! -d ${CONDA_PREFIX}/include/suitesparse ];
then
    ln -s ${CONDA_PREFIX}/include/ ${CONDA_PREFIX}/include/suitesparse
fi
make python
cp build/libxerus_misc.so ${CONDA_PREFIX}/lib/
cp build/libxerus.so ${CONDA_PREFIX}/lib/
# cp build/python3/xerus.so ${XERUS}
python -m pip install . --no-deps -vv

cd ../..
rm -rf ${TEMPDIR}
