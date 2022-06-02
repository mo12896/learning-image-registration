#!/bin/bash
git clone https://github.com/mo12896/ImageRegistration.git
cd ImageRegistration
conda create -n ir python=3.6
conda activate ir
pip install --upgrade pip
pip install -r requirements.txt