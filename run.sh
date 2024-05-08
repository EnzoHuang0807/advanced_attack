# python3 targeted_attack.py --GPU_ID 7 --steps 10
# python3 targeted_attack.py --GPU_ID 7 --steps 100
# python3 targeted_attack.py --GPU_ID 7 --steps 500
# python3 targeted_attack.py --GPU_ID 7 --steps 1000

python3 universal_attack.py --GPU_ID 7 --method SPGD  --eval 
python3 universal_attack.py --GPU_ID 7 --method SGA --minibatch 10 --eval 
# python3 universal_attack.py --GPU_ID 7 --method SGA --minibatch 20 --eval 
# python3 universal_attack.py --GPU_ID 7 --method SGA --minibatch 50 --eval
# python3 universal_attack.py --GPU_ID 7 --method SGA --minibatch 100 --eval  