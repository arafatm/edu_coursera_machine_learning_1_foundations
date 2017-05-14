import graphlab

with open('gl.key', 'r') as myfile:
    key = myfile.read()

print "read key %s" % (key)

graphlab.product_key.set_product_key(key)
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
graphlab.product_key.get_product_key()

sf = graphlab.SFrame('people-example.csv') # Load a tabular data set

sf # we can view first few lines of table

sf.tail() # view end of the table
          # .show() visualizes any data structure in GraphLab Create

