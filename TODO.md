1) Change all NDArray(...,...,...) so that shape is corrctly specified)
1a) Ensure that Shape() is specified such that exec can handle single
    as well as multiple NDArray inputs
2) check policy.sample() as referenced by dynamics.cpp
3) change trajSegment["mean"] so that it includes lqr feedback term
4) move trajSegment to be PolicyFunction variable
