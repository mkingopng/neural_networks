python3 kuzu_main.py --net lin  # problem

# missing one?

python3 kuzu_main.py --net conv  # ok

python3 spiral_main.py --net polar --hid 10  # ok

python3 spiral_main.py --net raw  # problem

python3 encoder_main.py --target=star16  #  ok

python3 encoder_main.py --target=input --dim=9 --plot  # works but annoying

python3 encoder_main.py --target=heart18  # ok

# something of your own design
python3 encoder_main.py --target=target1  # star

python3 encoder_main.py --target=target2  # double heart

# need something better than this
