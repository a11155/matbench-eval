nohup python test_sevennet.py --gpu 0 --left 0 --right 31250 > eval1_0_31250.log 2>&1 &
nohup python test_sevennet.py --gpu 1 --left 31250 --right 62500 > eval1_31250_62500.log 2>&1 &
nohup python test_sevennet.py --gpu 2 --left 62500 --right 93750 > eval1_62500_93750.log 2>&1 &
nohup python test_sevennet.py --gpu 3 --left 93750 --right 125000 > eval1_93750_125000.log 2>&1 &
nohup python test_sevennet.py --gpu 4 --left 125000 --right 156250 > eval1_125000_156250.log 2>&1 &
nohup python test_sevennet.py --gpu 5 --left 156250 --right 187500 > eval1_156250_187500.log 2>&1 &
nohup python test_sevennet.py --gpu 6 --left 187500 --right 218750 > eval1_187500_218750.log 2>&1 &
nohup python test_sevennet.py --gpu 7 --left 218750 --right -1 > eval1_218750_250000.log 2>&1 &
