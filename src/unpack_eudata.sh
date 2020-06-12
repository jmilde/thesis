#!/bin/sh

for filename in ./data/eudata/*; do unzip "$filename/*.zip" -d "./data/eudata_unpacked"; done
