% Dynamics model from G. Gremillion, S. Humbert paper "System Identification of
% a Quadrotor Micro Air Vehicle" (equation 3 in paper)
% Augmented to include position dynamics in hover
% Dynamics and controls matrix:
% cos(ps)*u  -sin(ps)*v    0     0     0     0     0     0     0     0     0     0     0     0     0     0
% sin(ps)*u  cos(ps)*v     0     0     0     0     0     0     0     0     0     0     0     0     0     0
%  0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0
%  0     0     0   X_u*u   0     0     0     0     0     0  X_th*th  0     0     0     0     0
%  0     0     0     0   Y_v*v   0     0     0     0  Y_ph*ph  0     0     0     0     0     0
%  0     0     0     0     0   Z_w*w   0     0     0     0     0     0     0     0     0 Z_thr*d_thr
%  0     0     0     0     0     0   L_p*p   0     0  L_ph*ph  0     0 L_la*d_la 0     0     0
%  0     0     0     0     0     0     0   M_q*q   0     0   M_th*th 0     0 M_lo*d_lo 0     0
%  0     0     0     0     0     0     0     0   N_r*r   0     0     0     0     0 N_ya*d_ya 0
%  0     0     0     0     0     0   ph_p*p  0     0     0     0     0 ph_la*d_la 0    0     0
%  0     0     0     0     0     0     0   th_q*q  0     0     0     0     0 th_lo*d_lo 0    0
%  0     0     0     0     0     0     0     0   ps_r*r  0     0     0     0     0 ps_ya*d_ya 0
%
%  Parameters:
%  X_u   =  -0.27996
%  Y_v   =  -0.22566
%  Z_w   =  -1.2991
%  L_p   =  -2.5110
%  M_q   =  -2.4467
%  N_r   =  -0.4948
%  X_th  =  -10.067
%  Y_ph  =   9.8648
%  L_ph  =  -21.358
%  M_th  =  -18.664
%  ph_p  =   0.9655
%  th_q  =   0.9634
%  ps_r  =   0.6748
%  Z_thr =  -39.282
%  L_la  =   11.468
%  M_lo  =   9.5711
%  N_ya  =   3.5647
%  ph_la =   0.0744
%  th_lo =   0.0594
%  ps_ya =   0.0397
% 
% State vector:
%    x y z u v w p q r phi theta psi
%
% Control vector:
%    del_lat del_lon del_yaw del_thrust

X_u   =  -0.27996;
Y_v   =  -0.22566;
Z_w   =  -1.2991;
L_p   =  -2.5110;
M_q   =  -2.4467;
N_r   =  -0.4948;
X_th  =  -10.067;
Y_ph  =   9.8648;
L_ph  =  -21.358;
M_th  =  -18.664;
ph_p  =   0.9655;
th_q  =   0.9634;
ps_r  =   0.6748;
Z_thr =  -39.282;
L_la  =   11.468;
M_lo  =   9.5711;
N_ya  =   3.5647;
ph_la =   0.0744;
th_lo =   0.0594;
ps_ya =   0.0397;

% Linearized about zero yaw
A = [ 0   0   0   1   0   0   0   0   0   0   0   0   ;
      0   0   0   0   1   0   0   0   0   0   0   0   ;
      0   0   0   0   0   1   0   0   0   0   0   0   ;
      0   0   0  X_u  0   0   0   0   0   0  X_th 0   ;
      0   0   0   0  Y_v  0   0   0   0  Y_ph 0   0   ;
      0   0   0   0   0  Z_w  0   0   0   0   0   0   ;
      0   0   0   0   0   0  L_p  0   0  L_ph 0   0   ;
      0   0   0   0   0   0   0  M_q  0   0  M_th 0   ;
      0   0   0   0   0   0   0   0  N_r  0   0   0   ;
      0   0   0   0   0   0  ph_p 0   0   0   0   0   ;
      0   0   0   0   0   0   0  th_q 0   0   0   0   ;
      0   0   0   0   0   0   0   0  ps_r 0   0   0   ];

B = [ 0   0   0   0   ;
      0   0   0   0   ;
      0   0   0   0   ;
      0   0   0   0   ;
      0   0   0   0   ;
      0   0   0 Z_thr ;
     L_la 0   0   0   ;
      0  M_lo 0   0   ;
      0   0  N_ya 0   ;
    ph_la 0   0   0   ;
      0 th_lo 0   0   ;
      0   0 ps_ya 0   ];

%TODO LQR controller
C = eye(12);
N = eye(12,4);
R = eye(4);
Q = eye(12);

% [K]:
%    0.0020    1.0000   -0.0000   -0.0418    1.3856   -0.0000    1.1039    0.0001   -0.0000    4.6346    0.0395   -0.0000
%   -1.0000    0.0020   -0.0043   -1.3583   -0.0458   -0.0015    0.0004    1.1348   -0.0028   -0.0247    4.6926   -0.0028
%   -0.0035    0.0000    0.1773   -0.0070   -0.0001   -0.0188   -0.0000   -0.0010    1.0131   -0.0002   -0.0142    0.9842
%    0.0037   -0.0000   -0.9842    0.0147    0.0001   -0.9920    0.0000    0.0044    0.2049    0.0001    0.2483    0.1773

[K,S,e] = lqr(A,B,Q,R,N);

H = ss(A-B*K,B,C,0);
G = ss(A,B,C,0);

t=0:.1:10;
t=t';
u=zeros(size(t,1),4);
x0=ones(1,12);
[y,t,x] = lsim(H,u,t,x0);

csvwrite('./data/lqrTest.csv',[t,y,u]);
