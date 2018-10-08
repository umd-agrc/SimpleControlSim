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
prefixPath['ndarrays'] = os.path.join('..','data','policy','arr2')
prefixPath['symbols'] = os.path.join('..','data','policy','sym')
prefixPath['lqr_ndarrays'] = os.path.join('..','data','lqr','arr')
prefixPath['ppo_ndarrays'] = os.path.join('..','data','policy','arr3')

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

    ndarrays['lqr_observation'] =parseNDArrayFile(
        os.path.join(prefixPath['lqr_ndarrays'],'observation.ndarray'))
    ndarrays['lqr_observation'] =\
        np.reshape(ndarrays['lqr_observation'],
                   (int(len(ndarrays['lqr_observation'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['ppo_observation0'] =parseNDArrayFile(
        os.path.join(prefixPath['ppo_ndarrays'],'observation-epoch0.ndarray'))
    ndarrays['ppo_observation0'] =\
        np.reshape(ndarrays['ppo_observation0'],
                   (int(len(ndarrays['ppo_observation0'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['ppo_observation10'] =parseNDArrayFile(
        os.path.join(prefixPath['ppo_ndarrays'],'observation-epoch10.ndarray'))
    ndarrays['ppo_observation10'] =\
        np.reshape(ndarrays['ppo_observation10'],
                   (int(len(ndarrays['ppo_observation10'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['ppo_observation20'] =parseNDArrayFile(
        os.path.join(prefixPath['ppo_ndarrays'],'observation-epoch20.ndarray'))
    ndarrays['ppo_observation20'] =\
        np.reshape(ndarrays['ppo_observation20'],
                   (int(len(ndarrays['ppo_observation20'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))



    ndarrays['observation0'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch0.ndarray'))
    ndarrays['observation0'] =\
        np.reshape(ndarrays['observation0'],
                   (int(len(ndarrays['observation0'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation2'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch2.ndarray'))
    ndarrays['observation2'] =\
        np.reshape(ndarrays['observation2'],
                   (int(len(ndarrays['observation2'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation4'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch4.ndarray'))
    ndarrays['observation4'] =\
        np.reshape(ndarrays['observation4'],
                   (int(len(ndarrays['observation4'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation6'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch6.ndarray'))
    ndarrays['observation6'] =\
        np.reshape(ndarrays['observation6'],
                   (int(len(ndarrays['observation6'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    '''
    ndarrays['observation8'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch8.ndarray'))
    ndarrays['observation8'] =\
        np.reshape(ndarrays['observation8'],
                   (int(len(ndarrays['observation8'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation10'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch10.ndarray'))
    ndarrays['observation10'] =\
        np.reshape(ndarrays['observation10'],
                   (int(len(ndarrays['observation10'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation12'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch12.ndarray'))
    ndarrays['observation12'] =\
        np.reshape(ndarrays['observation12'],
                   (int(len(ndarrays['observation12'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation14'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch14.ndarray'))
    ndarrays['observation14'] =\
        np.reshape(ndarrays['observation14'],
                   (int(len(ndarrays['observation14'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation16'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch16.ndarray'))
    ndarrays['observation16'] =\
        np.reshape(ndarrays['observation16'],
                   (int(len(ndarrays['observation16'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation18'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch18.ndarray'))
    ndarrays['observation18'] =\
        np.reshape(ndarrays['observation18'],
                   (int(len(ndarrays['observation18'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation20'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch20.ndarray'))
    ndarrays['observation20'] =\
        np.reshape(ndarrays['observation20'],
                   (int(len(ndarrays['observation20'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation22'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch22.ndarray'))
    ndarrays['observation22'] =\
        np.reshape(ndarrays['observation22'],
                   (int(len(ndarrays['observation22'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    ndarrays['observation24'] = parseNDArrayFile(
        os.path.join(prefixPath['ndarrays'],'observation-epoch24.ndarray'))
    ndarrays['observation24'] =\
        np.reshape(ndarrays['observation24'],
                   (int(len(ndarrays['observation24'])/OBSERVATION_LEN),
                    OBSERVATION_LEN))
    '''

    ndarrays['epoch'] = np.array(range(len(ndarrays['observation0'])))/12.5
    ndarrays['lqr_epoch'] = np.array(range(len(ndarrays['lqr_observation'])))/12.5
    ndarrays['ppo_epoch'] = np.array(range(len(ndarrays['ppo_observation0'])))/12.5

    p = plt.figure(title="Vehicle x velocity", tools=PLT_TOOLS)
    p.title.text_font_size = '20pt'
    p.legend.label_text_font_size = '15pt'
    p.legend.location = 'bottom_right'
    p.xaxis.axis_label = 'Time [s]'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label = 'Position [m]'
    p.yaxis.axis_label_text_font_size = '15pt'
    p.yaxis.major_label_text_font_size = '12pt'

    p.square(ndarrays['lqr_epoch'][0:100],ndarrays['lqr_observation'][0:100,18],
            color = '#000000', legend = 'lqr')
    p.triangle(ndarrays['ppo_epoch'],ndarrays['ppo_observation0'][:,18],
            color = '#320000', legend = 'ppo update 0')
    p.triangle(ndarrays['ppo_epoch'],ndarrays['ppo_observation10'][:,18],
            color = '#228b22', legend = 'ppo update 100')
    p.triangle(ndarrays['ppo_epoch'],ndarrays['ppo_observation20'][:,18],
            color = '#323200', legend = 'ppo update 300')

    p.scatter(ndarrays['epoch'],ndarrays['observation0'][:,18],
            color = '#fa3232', legend = 'aug update 0')
    p.scatter(ndarrays['epoch'],ndarrays['observation2'][:,18],
            color = '#3296fa', legend = 'aug update 100')
    p.scatter(ndarrays['epoch'],ndarrays['observation6'][:,18],
            color = '#329664', legend = 'aug update 300')
    plt.show(p)

if plotSymbols:
    pass

