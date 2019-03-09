from data import *

gen = DataGenerator()
gen.shrink_images((4,4))
# gen.featurize()
gen.export('data/4')
