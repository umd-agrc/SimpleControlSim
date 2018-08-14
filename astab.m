tspan = [0 10];
y0 = 100;
[t,y] = ode45(@(t,y) -100*y, tspan, y0);
csvwrite('./data/RK_AStabilityTestMatlab.csv',[t,y]);
