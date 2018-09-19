using DataFrames, Query, Compat, GLM

function receiveLoop!(port,dataQueue,ctl)
  # Connect to server
  cli = connect(port)
  try
    while !ctl["shouldExit"]
      push!(dataQueue,readline(cli))
      sleep(0.1)
    end
  catch ex
     
  finally
    close(cli)
  end
end

function testServer(port,ctl)
  shouldExit = false
  # Open server for listening
  server = listen(port)
  try
    while !ctl["shouldExit"]
      sock = accept(server)
      while isopen(sock)
        println(sock,"hi there\n")
      end
      sleep(0.5)
    end
  catch ex
    
  finally
    close(server)
  end
end

function parseMsg(msg)
    println(msg)
end

###################################################

portnum = 5000
dataQueue = []
#threads = Dict("server"=>Dict{Any,Any}("ctl"=>Dict("shouldExit"=>false)),
#               "client"=>Dict{Any,Any}("ctl"=>Dict("shouldExit"=>false)))

#threads["server"]["thread"] = @async testServer(portnum,threads["server"]["ctl"])

threads = Dict("client"=>Dict{Any,Any}("ctl"=>Dict("shouldExit"=>false)))
threads["client"]["thread"] = @async receiveLoop!(portnum,
                                                  dataQueue,
                                                  threads["client"]["ctl"])

shouldExit = false
try
  while !shouldExit
    if !isempty(dataQueue)
      el = pop!(dataQueue)
      parseMsg(el)
    end
    sleep(0.1) 
  end
catch ex
  #threads["server"]["ctl"]["shouldExit"] = true
  threads["client"]["ctl"]["shouldExit"] = true
end
