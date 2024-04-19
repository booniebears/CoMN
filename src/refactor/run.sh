if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j8
# ./main --PPA_cost tcad 1