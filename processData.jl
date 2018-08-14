using DataFrames, Query, Compat, GLM, CSV
using PlotlyJS

function readLqrData(filename)
  data = Float64[]
  label = Float64[]
  df = CSV.read(filename; delim=',', header=1)
  return df
end

dfMat = CSV.read("./data/lqrMatlab.csv";delim=",",header=1)
dfSim = CSV.read("./data/lqrCtl.csv";delim=",",header=1)
dfAStabMat = CSV.read("./data/RK_AStabilityTestMatlab.csv";delim=",",header=1)
dfAStabSim = CSV.read("./data/RK_AStabilityTest.csv";delim=",",header=1)

#=
trace1 = scatter(;x=dfMat[:t], y=dfMat[:u],
									mode="markers",
                  name="Matlab")
=#
trace2 = scatter(;x=dfSim[:t], y=dfSim[:u],
								  mode="markers",
                  name="Simulator") 

p1 = plot([trace2],Layout(xaxis_range=[0,10]))

#=
trace3 = scatter(;x=dfAStabMat[:t],y=dfAStabMat[:x],
									mode="markers",
									name="Matlab")
trace4 = scatter(;x=dfAStabSim[:t],y=dfAStabSim[:x],
									mode="markers",
									name="Simulator")

p2 = plot([trace3,trace4],Layout(xaxis_range=[0,5]))

[p1;p2]

# FIXME temporary training data mse estimate
dfMse = CSV.read("./data/sample_mse.csv";delim=",",header=1)
trace5 = scatter(;x=dfMse[:epoch], y=dfMse[:mse],
								  mode="markers",
                  name="Simulator") 

p2 = plot([trace5])
=#
