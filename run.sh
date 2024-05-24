#!/bin/bash

# Check if command-line arguments is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <model> <dataset>"
  exit 1
fi

# Get the command-line arguments
model=$1
dataset=$2

if [ "$model" != "gnb" ] && [ "$model" != "logit" ]; then
  echo "Invalid classification model: $model"
  exit 1
fi

# Execute your Python program based on the dataset
if [ "$dataset" == "breast" ]; then
  echo "Executing Python program with the breast dataset"
  python3 main.py $model breast.csv 0 31
elif [ "$dataset" == "wine" ]; then
  echo "Executing Python program with the wine dataset"
  python3 main.py $model wine.csv 11 12
elif [ "$dataset" == "rice" ]; then
  echo "Executing Python program with the rice dataset"
  python3 main.py $model rice.csv 7 8
elif [ "$dataset" == "letter" ]; then
  echo "Executing Python program with the letter dataset"
  python3 main.py $model letter.csv 0 17
elif [ "$dataset" == "magic" ]; then
  echo "Executing Python program with the magic dataset"
  python3 main.py $model magic.csv 10 11
else
  echo "Invalid dataset: $dataset"
  exit 1
fi
