#!/bin/bash


#  Installation of tropicalwidth package. 

if [ -e tropicalwidth-1.0.0.tar ]
then
  export ufiles=`find tropicalwidth -type f -mnewer tropicalwidth-1.0.0.tar | grep -v __pycache__`
  if [[ $ufiles != "" ]]
  then
    tar uvf tropicalwidth-1.0.0.tar $ufiles
  fi
else
  tar cvf tropicalwidth-1.0.0.tar `find tropicalwidth -type f | grep -v __pycache__`
fi

#  Installation of Miniconda. 

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  mfile="Miniconda3-py39_25.1.1-1-Linux-aarch64.sh"
else 
  mfile="Miniconda3-py39_25.1.1-1-Linux-x86_64.sh"
fi
if [[ ! -e $mfile ]]; then
  curl https://repo.anaconda.com/miniconda/$mfile -o $mfile
fi

rm -fr miniconda3.sh
ln $mfile miniconda3.sh

docker build -t rd-tropical-width .
docker tag rd-tropical-width:latest 939314430087.dkr.ecr.us-east-1.amazonaws.com/rd-tropical-width:latest
