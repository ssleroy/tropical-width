#!/bin/bash 

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 939314430087.dkr.ecr.us-east-1.amazonaws.com
docker push 939314430087.dkr.ecr.us-east-1.amazonaws.com/rd-tropical-width:latest
