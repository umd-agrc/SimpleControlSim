using MechanismGeometries
using RigidBodyDynamics
const rbd = RigidBodyDynamics
using GeometryTypes
using StaticArrays: SVector, SDiagonal
using CoordinateTransformations: AffineMap, IdentityTransformation, LinearMap, Translation
using ColorTypes: RGBA
using MeshCat, RigidBodyDynamics, MeshCatMechanisms

function homog(::IdentityTransformation)
  eye(4)
end

function homog(m::AffineMap)
  CoordinateTransformations.Transformation(m)
end

homog(t::Translation) = homog(AffineMap(LinearMap(eye(3)),t.v))
homog(r::LinearMap,t::Translation) = homog(AffineMap(r.m,t.v))

vis = Visualizer()
urdf = joinpath(pwd(), "resources", "quadrotor.urdf")
robot = parse_urdf(Float64, urdf)
delete!(vis)
mvis = MechanismVisualizer(robot, URDFVisuals(urdf), vis)
body = bodies(robot)[end]
body_frame = default_frame(body)
setelement!(mvis,body_frame)
at = homog(Translation(0,0,-1))
settransform!(vis,homog(LinearMap(eye(3)),Translation(0,0,0)))
open(vis)
for i=1:1000
  settransform!(vis,homog(LinearMap(CoordinateTransformations.RotX(i/10)),Translation(0,0,0)))
  sleep(0.05)
end
