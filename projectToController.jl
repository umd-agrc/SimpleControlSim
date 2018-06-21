using DataFrames, Query, Compat, GLM, CSV
using PlotlyJS

function percentError(estimate,exact)
  return abs((estimate-exact)/exact)*100
end

function percentDifference(val1,val2)
  return abs((val1-val2)/((val1+val2)/2))*100
end

function readTestData(f)
  re = r"^\[(?<data>[-?\d+\.\d+\s*]+)\],\s+\[(?<label>[-?\d+\.\d+?\s*]+)\]$"
  data = Float64[]
  label = Float64[]
  # Read headerline
  header = readline(f)
  while !eof(f)
    s = readline(f)
    m = match(re,s)
    data = [data;readdlm(IOBuffer(m[:data]))]
    label = [label;readdlm(IOBuffer(m[:label]))]
  end
  data = convert(DataFrame,
                 Dict(zip(["data$i" for i in 1:size(data,2)],
                          [data[:,i] for i in 1:size(data,2)])))
  label = convert(DataFrame,
                  Dict(zip(["label$i" for i in 1:size(label,2)],
                           [label[:,i] for i in 1:size(label,2)])))
  return hcat(data,label)
end

function readLqrData(filename)
  data = Float64[]
  label = Float64[]
  df = CSV.read(filename; delim=',', header=1)
  return df
end

function getQuadAngleControl(df)
  controls = Float64[]
  controlSystem = [1 1 -1 -1;
                   1 -1 -1 1;
                   1 1 1 -1;
                   1 -1 1 1]

  for i=1:size(df,1)
    controls = [controls ;
      transpose(controlSystem \
                [df[:label1][i],df[:label2][i],df[:label3][i],df[:label4][i]])]
  end
  controls = convert(DataFrame,
                     Dict(zip(["thrustCtl" "pitchCtl" "rollCtl" "yawCtl"],
                              [controls[:,i] for i in 1:size(controls,2)])))

  return hcat(df,controls)
end

#########################################

token = "LQR"
if token == "test"
  f = open("./data/test.txt","r")

  df = readTestData(f);
  df = getQuadAngleControl(df)

  linearModel = Dict("pitch" => lm(@formula(pitchCtl ~ data10 + data11 + data12 +
                                            data4 + data5 + data6), df),
                     "roll" => lm(@formula(rollCtl ~ data13 + data14 + data15 +
                                           data1 + data2 + data3), df),
                     "yaw" => lm(@formula(yawCtl ~ data7 + data8 + data9),df),
                     "thrust" => lm(@formula(thrustCtl ~ data16 + data17 + data18),df))

  print(linearModel["roll"])
  print(linearModel["yaw"])
  print(linearModel["thrust"])

elseif token == "PID"

elseif token == "LQR"
 
  # LQR gains
  gains = [0.0020 1.0000 -0.0000 -0.0418 1.3856 -0.0000 1.1039 0.0001 -0.0000 4.6346 0.0395 -0.0000;
           -1.0000 0.0020 -0.0043 -1.3583 -0.0458 -0.0015 0.0004 1.1348 -0.0028 -0.0247 4.6926 -0.0028;
           -0.0035 0.0000 0.1773 -0.0070 -0.0001 -0.0188 -0.0000 -0.0010 1.0131 -0.0002 -0.0142 0.9842;
           0.0037 -0.0000 -0.9842 0.0147 0.0001 -0.9920 0.0000 0.0044 0.2049 0.0001 0.2483 0.1773]

  f = Dict{Any,Any}()
  df = Dict{Any,Any}()
  gainDiff = Dict{Any,Any}()
  linearModels = Dict{Any,Dict{String,Any}}()
  f[1] = [0,"data/lqrNn0Loops.csv"]
  f[2] = [5,"data/lqrNn5Loops.csv"]
  f[3] = [10,"data/lqrNn10Loops.csv"]
  f[4] = [15,"data/lqrNn15Loops.csv"]
  f[5] = [20,"data/lqrNn20Loops.csv"]

  for i=1:5
    numLoops = f[i][1]
    df[numLoops] = readLqrData(f[i][2])
    linearModels[i] = Dict("del_lat" => lm(@formula(del_lat ~ x + y + z + u + v
                                                            + w + p + q + r +
                                                            phi + theta + psi),
                                           df[numLoops]),
                           "del_lon" => lm(@formula(del_lon ~ x + y + z + u + v
                                                            + w + p + q + r +
                                                            phi + theta + psi),
                                           df[numLoops]),
                           "del_yaw" => lm(@formula(del_yaw ~ x + y + z + u + v
                                                            + w + p + q + r +
                                                            phi + theta + psi),
                                           df[numLoops]),
                           "del_thrust" => lm(@formula(del_thrust ~ x + y + z + u + v
                                                                  + w + p + q + r +
                                                                  phi + theta +
                                                                  psi),df[numLoops]))

    gainDiff[numLoops] = zeros(size(gains))
    r=[]

    push!(r,[coef(linearModels[i]["del_lat"])[2:end] gains[1,:]])
    push!(r,[coef(linearModels[i]["del_lon"])[2:end] gains[2,:]])
    push!(r,[coef(linearModels[i]["del_yaw"])[2:end] gains[3,:]])
    push!(r,[coef(linearModels[i]["del_thrust"])[2:end] gains[4,:]])

    for i=1:4
      for j=1:size(r[i],1)
        gainDiff[numLoops][i,j] = percentDifference(r[i][j,1],r[i][j,2])
      end
    end
  end

  numLoopsSet = [0 5 10 15 20]'
  gainDiffChange = [mean(gainDiff[i]) for i in numLoopsSet]'

  # Plotting
  #=
  trace1 = scatter(;x=numLoopsSet,y=gainDiffChange,
            mode="markers",
            marker=attr(color="#02010f",size=10,symbol="square",
                        line=attr(color="#02010f",width=0)),
            name="Gain Percent Difference")

  layout = Layout(autosize=false, width=500, height=500,
                  margin=attr(l=0,r=0,b=0,t=65))
  p = plot([trace1],layout)
  =#

  plot(scatter(x=numLoopsSet,y=gainDiffChange,
               marker=attr(color="#b3010f",size=10,symbol="square",
                           line=attr(color="#02010f",width=0)),
               name="Gain Percent Difference"))
  display(p)

end


