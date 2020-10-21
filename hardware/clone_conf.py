def clone_conf():
    fa = open('config/1_1config.txt', 'r')
    fb = open('config/1_2config.txt', 'w')
    fc = open('config/1_3config.txt', 'w')
    fr = open("config/re_config.txt", 'r')
    frA = open("config/re_config_A.txt", 'w')
    frB = open("config/re_config_B.txt", 'w')
    frC = open("config/re_config_C.txt", 'w')

    configA = fa.readlines()

    for ins in configA:

        ins = ins.replace('4', '8', 1)
        fb.write(ins)
        ins = ins.replace('8', 'C', 1)
        fc.write(ins)

    for ins in fr:
        frA.write(ins)
        ins = ins.replace('4', '8', 1)
        frB.write(ins)
        ins = ins.replace('8', 'C', 1)
        frC.write(ins)