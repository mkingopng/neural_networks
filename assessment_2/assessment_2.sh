python3 kuzu_main.py --net lin  # ok

python3 kuzu_main.py --net full  # ok

python3 kuzu_main.py --net conv  # ok

python3 spiral_main.py --net polar --hid 10  # ok

python3 spiral_main.py --net raw  # ok

python3 encoder_main.py --target=star16  # ok

python3 encoder_main.py --target=input --dim=9 --plot  #

python3 encoder_main.py --target=heart18  # ok

# something of your own design
python3 encoder_main.py --target=target1  # smile

python3 encoder_main.py --target=target2  # boat

# need something better than this
