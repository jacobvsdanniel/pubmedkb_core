#!/usr/bin/env bash

# Start GNormPlusServer
cd ~/bern/GNormPlusJava
nohup java -Xmx16G -Xms16G -jar GNormPlusServer.jar 18895 >> ~/bern/logs/nohup_gnormplus.out 2>&1 &

# Start tmVar2Server
cd ~/bern/tmVarJava
nohup java -Xmx8G -Xms8G -jar tmVar2Server.jar 18896 >> ~/bern/logs/nohup_tmvar.out 2>&1 &

# Start normalizers
cd ~/bern/
sh load_dicts.sh

# Set your GPU number(s)
export CUDA_VISIBLE_DEVICES=0

# Run BERN
nohup python3 -u server.py --port 8888 --gnormplus_home ~/bern/GNormPlusJava --gnormplus_port 18895 --tmvar2_home ~/bern/tmVarJava --tmvar2_port 18896 >> logs/nohup_BERN.out 2>&1 &

# sleep infinity

