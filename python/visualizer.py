import os
import re
import bokeh.plotting as plt
import numpy as np
import numpy.linalg as nplin

def parseNDArrayFile(filepath):
    d=None
    with open(filepath) as f:
        i=0
        for line in map(str.strip,f):
            if i==0:
                i = i + 1
                continue
            l = re.split(',\s*(?![^()[]]*[\)\]])',re.sub('[\[\]\s+]','',line).strip(','))
            d = np.array([float(x) for x in l])
            i = i + 1
    return d

PLT_TOOLS='hover,crosshair,pan,wheel_zoom,box_zoom,reset,tap,save,box_select,poly_select,lasso_select'

#TODO fix to plot all available
plotExecutionTimes = False
plotArrays = True
plotSymbols = False
p=None

prefixPath={}
prefixPath['executionTimes'] = os.path.join('..','data','policy','dbg','execution_times')
prefixPath['ndarrays'] = os.path.join('..','data','policy','arr')
prefixPath['symbols'] = os.path.join('..','data','policy','sym')

executionTimes={}
ndarrays={}
symbols={}

if plotExecutionTimes:
    executionTimes["policySingleForwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'policySingleForwardPass.ndarray'))
    executionTimes["policyBatchBackwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'policyBatchBackwardPass.ndarray'))
    executionTimes["valueBatchForwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'valueBatchForwardPass.ndarray'))
    executionTimes["valueBatchBackwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'valueBatchBackwardPass.ndarray'))
    executionTimes["lossBatchForwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'lossBatchForwardPass.ndarray'))
    executionTimes["lossBatchBackwardPass"] = parseNDArrayFile(
            os.path.join(prefixPath['executionTimes'],'lossBatchBackwardPass.ndarray'))

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

    p = plt.figure(title="Mean time to execute", tools=PLT_TOOLS)
    bar_opts = dict(width=0.7,alpha=0.8)
    for t in meanExecutionTimes:
        p.vbar(bottom = 0, top = meanExecutionTimes[t],
                x = meanExecutionTimesIdx[t], color = colors[t], legend = t, **bar_opts)
    plt.show(p)

if plotArrays:
    OBSERVATION_LEN = 24
    ndarrays['observation0'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch0.ndarray'))
    ndarrays['observation0'] =\
        np.reshape(ndarrays['observation0'],
                   (int(len(ndarrays['observation0'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation80'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch80.ndarray'))
    ndarrays['observation80'] =\
        np.reshape(ndarrays['observation80'],
                   (int(len(ndarrays['observation80'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation160'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch160.ndarray'))
    ndarrays['observation160'] =\
        np.reshape(ndarrays['observation160'],
                   (int(len(ndarrays['observation160'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation240'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch240.ndarray'))
    ndarrays['observation240'] =\
        np.reshape(ndarrays['observation240'],
                   (int(len(ndarrays['observation240'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation320'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch320.ndarray'))
    ndarrays['observation320'] =\
        np.reshape(ndarrays['observation320'],
                   (int(len(ndarrays['observation320'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation400'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch400.ndarray'))
    ndarrays['observation400'] =\
        np.reshape(ndarrays['observation400'],
                   (int(len(ndarrays['observation400'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation480'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch480.ndarray'))
    ndarrays['observation480'] =\
        np.reshape(ndarrays['observation480'],
                   (int(len(ndarrays['observation480'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation560'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch560.ndarray'))
    ndarrays['observation560'] =\
        np.reshape(ndarrays['observation560'],
                   (int(len(ndarrays['observation560'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation640'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch640.ndarray'))
    ndarrays['observation640'] =\
        np.reshape(ndarrays['observation640'],
                   (int(len(ndarrays['observation640'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation720'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch720.ndarray'))
    ndarrays['observation720'] =\
        np.reshape(ndarrays['observation720'],
                   (int(len(ndarrays['observation720'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation800'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch800.ndarray'))
    ndarrays['observation800'] =\
        np.reshape(ndarrays['observation800'],
                   (int(len(ndarrays['observation800'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation880'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch880.ndarray'))
    ndarrays['observation880'] =\
        np.reshape(ndarrays['observation880'],
                   (int(len(ndarrays['observation880'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation960'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch960.ndarray'))
    ndarrays['observation960'] =\
        np.reshape(ndarrays['observation960'],
                   (int(len(ndarrays['observation960'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))

    ndarrays['epoch'] = np.array(range(len(ndarrays['observation0'])))/12.5

    p = plt.figure(title="Vehicle x position", tools=PLT_TOOLS)
    p.title.text_font_size = '20pt'
    p.legend.label_text_font_size = '15pt'
    p.xaxis.axis_label = 'Time [s]'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label = 'Position [m]'
    p.yaxis.axis_label_text_font_size = '15pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.scatter(ndarrays['epoch'],ndarrays['observation0'][:,12],
            color = '#e65c5c', legend = 'update 0')
    '''
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation80'][:,12:]),
            color = '#e6935c', legend = 'observation 80')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation160'][:,12:]),
            color = '#e6d15c', legend = 'observation 160')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation240'][:,12:]),
            color = '#bfe65c', legend = 'observation 240')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation320'][:,12:]),
            color = '#88e65c', legend = 'observation 320')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation400'][:,12:]),
            color = '#5ce67f', legend = 'observation 400')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation480'][:,12:]),
            color = '#5ce6d1', legend = 'observation 480')
    '''
    p.scatter(ndarrays['epoch'],ndarrays['observation560'][:,12],
            color = '#5cb3e6', legend = 'update 560')
    '''
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation640'][:,12:]),
            color = '#5c78e6', legend = 'observation 640')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation720'][:,12:]),
            color = '#785ce6', legend = 'observation 720')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation800'][:,12:]),
            color = '#c15ce6', legend = 'observation 800')
    p.scatter(ndarrays['epoch'],nplin.norm(ndarrays['observation880'][:,12:]),
            color = '#e65cca', legend = 'observation 880')
            '''
    p.scatter(ndarrays['epoch'],ndarrays['observation960'][:,12],
            color = '#e65c9c', legend = 'update 960')
    plt.show(p)

if plotSymbols:
    pass

