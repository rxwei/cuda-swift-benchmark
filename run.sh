#!/bin/bash

make

for f in `find .build/debug -perm +111 -type f`
do
    ${f}
done
