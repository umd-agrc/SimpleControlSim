using Distributions 

struct PIDController
  roll
  pitch
  yaw
  x
  y
  z
  trimThrust
end

struct LQRController
  gains 
end

struct PIDDataDistributions
  rollP
  rollI
  rollD
  pitchP
  pitchI
  pitchD
  yawP
  yawI
  yawD
  xP
  xI
  xD
  yP
  yI
  yD
  zP
  zI
  zD
end

struct LQRDataDistributions
  x
  y
  z
  u
  v
  w
  p
  q
  r
  phi
  theta
  psi
end

const NUM_DATAPOINTS = 1000
const MOTOR_MAX_PWM = 2000

function bound(lowerLim,upperLim,value)
  return max(min(value,upperLim),lowerLim)
end

function setController(rollPid,pitchPid,yawPid,xPid,yPid,zPid)
  return PIDController(rollPid,pitchPid,yawPid,xPid,yPid,zPid,900,590)
end

function setController(gains)
  return LQRController(gains)
end

function setDistributions(bw,distType)
  if distType == "PID"
    return PIDDataDistributions(Distributions.Uniform(bw["roll"][1]...),
                                Distributions.Uniform(bw["roll"][2]...),
                                Distributions.Uniform(bw["roll"][3]...),
                                Distributions.Uniform(bw["pitch"][1]...),
                                Distributions.Uniform(bw["pitch"][2]...),
                                Distributions.Uniform(bw["pitch"][3]...),
                                Distributions.Uniform(bw["yaw"][1]...),
                                Distributions.Uniform(bw["yaw"][2]...),
                                Distributions.Uniform(bw["yaw"][3]...),
                                Distributions.Uniform(bw["x"][1]...),
                                Distributions.Uniform(bw["x"][2]...),
                                Distributions.Uniform(bw["x"][3]...),
                                Distributions.Uniform(bw["y"][1]...),
                                Distributions.Uniform(bw["y"][2]...),
                                Distributions.Uniform(bw["y"][3]...),
                                Distributions.Uniform(bw["z"][1]...),
                                Distributions.Uniform(bw["z"][2]...),
                                Distributions.Uniform(bw["z"][3]...))
  elseif distType == "LQR"
     return LQRDataDistributions(Distributions.Uniform(bw["x"]...),
                                 Distributions.Uniform(bw["y"]...),
                                 Distributions.Uniform(bw["z"]...),
                                 Distributions.Uniform(bw["p"]...),
                                 Distributions.Uniform(bw["q"]...),
                                 Distributions.Uniform(bw["r"]...),
                                 Distributions.Uniform(bw["u"]...),
                                 Distributions.Uniform(bw["v"]...),
                                 Distributions.Uniform(bw["w"]...),
                                 Distributions.Uniform(bw["phi"]...),
                                 Distributions.Uniform(bw["theta"]...),
                                 Distributions.Uniform(bw["psi"]...))
  end
end

function generatePoints(controller::PIDController,
                        distributions::PIDDataDistributions,
                        numPoints)
  # Simulate measured data
  data = [transpose([rand(distributions.rollP),
                     rand(distributions.rollI),
                     rand(distributions.rollD),
                     rand(distributions.pitchP),
                     rand(distributions.pitchI),
                     rand(distributions.pitchD),
                     rand(distributions.yawP),
                     rand(distributions.yawI),
                     rand(distributions.yawD),
                     rand(distributions.xP),
                     rand(distributions.xI),
                     rand(distributions.xD),
                     rand(distributions.yP),
                     rand(distributions.yI),
                     rand(distributions.yD),
                     rand(distributions.zP),
                     rand(distributions.zI),
                     rand(distributions.zD)])
          for i=1:numPoints]
  # Determine control for measured data
  label = [pidControl(p,controller) for p in data]
  return Dict("data" => data,
              "label" => label)
end

function generatePoints(controller::LQRController,
                        distributions::LQRDataDistributions,
                        numPoints)
  data = [[[rand(distributions.x),
            rand(distributions.y),
            rand(distributions.z),
            rand(distributions.u),
            rand(distributions.v),
            rand(distributions.w),
            rand(distributions.p),
            rand(distributions.q),
            rand(distributions.r),
            rand(distributions.phi),
            rand(distributions.theta),
            rand(distributions.psi)];
           zeros(12,1)]
          for i=1:numPoints]
  data = map(p->transpose(p),data)
  label = [lqrControl(p,controller) for p in data]
  return Dict("data" => data,
              "label" => label)
end

#FIXME ensure correct
# Assumes motors as such: forward left, forward right, back left, back right
# Coordinate system:
#   positive x right, y back, z down
#   positive pitch back, roll left, yaw right
function pidControl(pt,controller::PIDController)
  rollDesired =  pt[13]*controller.x[1] +
                 pt[14]*controller.x[2] +
                 pt[15]*controller.x[3]
  pitchDesired = pt[10]*controller.y[1] +
                 pt[11]*controller.y[2] +
                 pt[12]*controller.y[3]

  pitchControl = (pt[4] - pitchDesired)*controller.pitch[1] + 
                 pt[5]*controller.pitch[2] + 
                 pt[6]*controller.pitch[3] 
  rollControl = (pt[1] - rollDesired)*controller.roll[1] + 
                 pt[2]*controller.roll[2] + 
                 pt[3]*controller.roll[3] 
  thrustControl = pt[16]*controller.z[1] +
                  pt[17]*controller.z[2] +
                  pt[18]*controller.z[3] +
                  controller.trimThrust +
                  controller.baseThrust
  yawControl = pt[7]*controller.yaw[1] + 
               pt[8]*controller.yaw[2] +
               pt[9]*controller.yaw[3]

  ff = open("./data/tmp.txt","a+")
  write(ff,"$thrustControl,$pitchControl,$rollControl,$yawControl\n")
  close(ff)

  lf = thrustControl+rollControl-pitchControl-yawControl
  rf = thrustControl-rollControl-pitchControl+yawControl
  lb = thrustControl+rollControl+pitchControl+yawControl
  rb = thrustControl-rollControl+pitchControl-yawControl
  return map(u->bound(-MOTOR_MAX_PWM,MOTOR_MAX_PWM,u),[u1 u2 u3 u4])
end

function lqrControl(pt,controller::LQRController)
  return controller.gains*pt[1:12]
end

function writePoints(f,p,controlType)
  data = p["data"]
  label = p["label"]
  if controlType == "PID"
    header = "[rollP rollI rollD pitchP pitchI pitchD yawP yawI yawD xP xI xD yP yI yD zP zI zD], [lf rf lb rb]\n"
  elseif controlType == "LQR"
    header = "[xd yd zd ud vd wd pd qd rd phid thetad psid x y z u v w p q r phi theta psi], [del_lat del_lon del_yaw del_thr]\n"
  end
  write(f,header)
  #for i in length(data) 
  #  writedlm(f, [data label], ", ")
  #end
  writedlm(f, [data label], ", ")
end

#########################################

token = "LQR"
if token == "test"
  f = open("./data/test.txt","w")

  rollPid = [-2,0,40*.017]
  pitchPid = [2,0,40*.017]
  yawPid = [23.22,.192,80*.017]
  xPid = [.0122,.0041,.026]
  yPid = [.0122,.0041,.026]
  zPid = [.2330,.069,.0188]

  controller = setController(rollPid,pitchPid,yawPid,xPid,yPid,zPid)

  rollBw = [[-pi pi],[-1000 1000],[-5 5]]
  pitchBw = [[-pi pi],[-1000 1000],[-5 5]]
  yawBw = [[-pi pi],[-1000 1000],[-5 5]]
  xBw = [[-2000 2000],[-4000 4000],[-3000 3000]]
  yBw = [[-2000 2000],[-4000 4000],[-3000 3000]]
  zBw = [[-2000 2000],[-4000 4000],[-3000 3000]]
  bw = Dict("roll" => rollBw,
            "pitch" => pitchBw,
            "yaw" => yawBw,
            "x" => xBw,
            "y" => yBw,
            "z" => zBw)

  distributions = setDistributions(bw,"PID")

  numPoints = NUM_DATAPOINTS

  points = generatePoints(controller,distributions,numPoints)
  writePoints(f,points,"PID")

  close(f)
elseif token == "LQR"
  f = open("./data/lqrData.txt","w")
  gains = [0.0020 1.0000 -0.0000 -0.0418 1.3856 -0.0000 1.1039 0.0001 -0.0000 4.6346 0.0395 -0.0000;
           -1.0000 0.0020 -0.0043 -1.3583 -0.0458 -0.0015 0.0004 1.1348 -0.0028 -0.0247 4.6926 -0.0028;
           -0.0035 0.0000 0.1773 -0.0070 -0.0001 -0.0188 -0.0000 -0.0010 1.0131 -0.0002 -0.0142 0.9842;
           0.0037 -0.0000 -0.9842 0.0147 0.0001 -0.9920 0.0000 0.0044 0.2049 0.0001 0.2483 0.1773]
  controller = setController(gains)

  xBw = [-5 5]
  yBw = [-5 5]
  zBw = [-5 5]
  uBw = [-5 5]
  vBw = [-5 5]
  wBw = [-5 5]
  pBw = [-7 7]
  qBw = [-7 7]
  rBw = [-7 7]
  phiBw = [-pi pi]
  thetaBw = [-pi pi]
  psiBw = [-pi pi]
  bw = Dict("x" => xBw,
            "y" => yBw,
            "z" => zBw,
            "u" => uBw,
            "v" => vBw,
            "w" => wBw,
            "p" => pBw,
            "q" => qBw,
            "r" => rBw,
            "phi" => phiBw,
            "theta" => thetaBw,
            "psi" => psiBw)

  distributions = setDistributions(bw,"LQR")

  numPoints = NUM_DATAPOINTS
  points = generatePoints(controller,distributions,numPoints)
  writePoints(f,points,"LQR")
  close(f)
elseif token == "PID"
end
