using GLVisualize, GeometryTypes, GLAbstraction, Colors, FileIO

window = glscreen()

mesh = GLNormalMesh(loadasset("./resources/quadrotor.obj")

view(visualize(mesh), window)

renderloop(window)

