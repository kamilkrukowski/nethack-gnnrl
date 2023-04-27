# nethack-gnnrl

a Graph-Neural-Network approach for reinforcement learning using the [NetHack Learning Environment](https://github.com/facebookresearch/nle)

## Replay `ttyrec3` Files

We modified `ttyplay.py` from the NLE repository (https://github.com/facebookresearch/nle) in lines 106 and 127 to allow for the 13th header to have a value of 2.

Example Usage:
```
python ttyplay.py nle.77538.3000.ttyrec3.bz2 -s 1 --fixed_frame_wait 
```