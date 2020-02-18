import numpy as np
import pandas as pd
import plumage.synthetic as synth

# Initialise
# 4010201828880793984
bcor = 16.7689
rv =  3.82773
#params = (3993, 4.67, 0.46)
params = (6000, 3, -2)

# 1243381938292426624
star_id = "1243381938292426624"
bcor = 25.0304
rv =  -4.08545
params = (4012, 1.22, -0.28)


# 6316829270922140032 (37 lib)
star_id = "6316829270922140032"
bcor = 29.664
rv = 48.921
params = (4813, 3.05, 0.02)

# 2452378776434276992 (Tau Cet)
star_id = "2452378776434276992"
bcor = 26.142
rv = -20.0515
params = (5353, 4.44, -0.52)

# 2611163717366876544, 2815543034682035840, 119186137034100736, 3209938366665770752
star_id = "2815543034682035840"
bcor = 9.29967
rv = -7.32428
params = (4013, 4.65, -0.05)

# 119186137034100736 (giant)
star_id = "119186137034100736"
bcor = 16.1337
rv = -5.21783
params = (4485, 1.29, -0.27)

# 6421542154150684160 (HR 7221)
star_id = "6421542154150684160"
bcor = -16.8851
rv = -9.77961
params = (5018, 3.49, -0.05)

tc_spec = np.loadtxt("spec_tx_%s.csv" % star_id)
#params = (6000, 4, -2)

xx = synth.do_synthetic_fit(
    tc_spec[0], 
    tc_spec[1], 
    tc_spec[2], 
    params, 
    rv, 
    bcor)
