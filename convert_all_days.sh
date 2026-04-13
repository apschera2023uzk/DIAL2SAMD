#!/bin/bash

for day in $(seq -w 1 31); do
    ./dial2samd.py -d 202408${day} 2>/dev/null || true
done
