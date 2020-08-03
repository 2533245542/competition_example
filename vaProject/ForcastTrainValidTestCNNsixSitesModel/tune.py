import math
import os

file = '663A4baseLSTMMultivariate.py'
# file = '663GAbaseLSTMMultivariate.py'
# file = '663GBbaseLSTMMultivariate.py'
# file = '663GCbaseLSTMMultivariate.py'
# file = '663GDbaseLSTMMultivariate.py'
# file = '663GEbaseLSTMMultivariate.py'
# file = '663baseLSTMMultivariate.py'

for i in [0]:
    f = open(file, 'r')
    lines = f.readlines()
    # lines[84] = "model.compile(optimizer=Adam(lr={}), loss='mae')\n".format(i)
    f.close()

    f = open(file, 'w')
    f.writelines(lines)
    f.close()
    os.system('python3 -W ignore ' + file)
    print('for', i)


# cd '../../vaProject/CNNsixSitesModel'

