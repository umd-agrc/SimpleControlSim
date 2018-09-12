import os
import re
import bokeh.plotting as plt
import numpy as np

def parseNDArrayFile(filepath):
    d=None
    with open(filepath) as f:
        i=0
        for line in map(str.strip,f):
            if i==0:
                i = i + 1
                continue
            l = re.split(',\s*(?![^()[]]*[\)\]])',re.sub('[\[\]\s+]','',line).strip(','))
            d = [float(x)*1e-6 for x in l]
            i = i + 1
    return d

prefixPath['executionTimes'] = os.path.join('..','data','policy','dbg','execution_times')
prefixPath['arrays'] = os.path.join('..','data','policy','arr')

executionTimes={}
executionTimes["policySingleForwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/policySingleForwardPass.ndarray")
executionTimes["policyBatchBackwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/policyBatchBackwardPass.ndarray")
executionTimes["valueBatchForwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/valueBatchForwardPass.ndarray")
executionTimes["valueBatchBackwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/valueBatchBackwardPass.ndarray")
executionTimes["lossBatchForwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/lossBatchForwardPass.ndarray")
executionTimes["lossBatchBackwardPass"] = parseNDArrayFile(
        "../data/policy/dbg/execution_times/lossBatchBackwardPass.ndarray")

meanExecutionTimes = {k : np.mean(executionTimes[k]) for k in executionTimes}
print(meanExecutionTimes) 
meanExecutionTimesIdx = {
        "policySingleForwardPass" : 1,
        "policyBatchBackwardPass" : 2,
        "valueBatchForwardPass" : 3,
        "valueBatchBackwardPass" : 4,
        "lossBatchForwardPass" : 5,
        "lossBatchBackwardPass" : 6
        }
colors = {
        "policySingleForwardPass" : '#800080',
        "policyBatchBackwardPass" : '#800000',
        "valueBatchForwardPass" : '#003366',
        "valueBatchBackwardPass" : '#008000',
        "lossBatchForwardPass" : '#ffa500',
        "lossBatchBackwardPass" : '#191919' 
        }

PLT_TOOLS='hover,crosshair,pan,wheel_zoom,box_zoom,reset,tap,save,box_select,poly_select,lasso_select'

p = plt.figure(title="Mean time to execute", tools=PLT_TOOLS)
bar_opts = dict(width=0.7,alpha=0.8)
for t in meanExecutionTimes:
    p.vbar(bottom = 0, top = meanExecutionTimes[t],
            x = meanExecutionTimesIdx[t], color = colors[t], legend = t, **bar_opts)
plt.show(p)
