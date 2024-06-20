#!/bin/bash

cmake --build build && ./build/$1 --test-case=$2