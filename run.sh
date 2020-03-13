# unet results
#python3 main.py exp datasets/textures1.pickle -b 5 -e 10 -mp 1 -w 10 -p 1 -c exp1 >> exp1.log
#python3 main.py exp datasets/textures2.pickle -b 5 -e 10 -mp 1 -w 10 -p 1 -c exp2 >> exp2.log
#python3 main.py exp datasets/textures3.pickle -b 5 -e 10 -mp 1 -w 10 -p 1 -c exp3 >> exp3.log

python3 main.py evo datasets/textures1.pickle -b 4 -e 10 -ps 10 -ev 1000 -mp 1 -w 4 -f evo1 >> evo1.log
python3 main.py evo datasets/textures2.pickle -b 4 -e 10 -ps 10 -ev 1000 -mp 1 -w 4 -f evo2 >> evo2.log
python3 main.py evo datasets/textures3.pickle -b 4 -e 10 -ps 10 -ev 1000 -mp 1 -w 4 -f evo3 >> evo3.log
