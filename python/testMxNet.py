import os
import os.path
import re
import bokeh.plotting as plt

filepath = '../data/test/test_mxnet/tmp.txt'

d={}
PLT_TOOLS='hover,crosshair,pan,wheel_zoom,box_zoom,reset,tap,save,box_select,poly_select,lasso_select'
with open(filepath) as f:
    i=0
    for line in map(str.strip,f):
        l = re.split(',\s*(?![^()[]]*[\)\]])',re.sub('[\[\]\s+]','',line).strip(','))
        d[i] = [float(x) for x in l]
        i = i + 1

p1 = plt.figure(width=600,height=600,tools=PLT_TOOLS)
p1.scatter(range(len(d[0])),d[0])
plt.show(p1)
