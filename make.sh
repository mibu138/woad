#!/usr/bin/env bash
if [ ! -f pome ]; then
    tar -xvf pome.tar.gz
fi
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 32
