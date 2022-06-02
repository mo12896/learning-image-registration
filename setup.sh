#!/bin/bash
conda create -n ir python=3.6
conda activate ir
pip install --upgrade pip
pip install -r requirements.txt