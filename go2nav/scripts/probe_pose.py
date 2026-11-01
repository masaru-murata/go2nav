"""Experiment: does a sport gesture work with obstacle avoidance OFF? (run on the Jetson, robot process may stay up)"""
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

action = sys.argv[1] if len(sys.argv) > 1 else "Hello"
ChannelFactoryInitialize(0, "eth0")
sc = SportClient(); sc.SetTimeout(5.0); sc.Init()
oa = ObstaclesAvoidClient(); oa.SetTimeout(5.0); oa.Init()
print("oa before:", oa.SwitchGet())
print("action with OA on ->", getattr(sc, action)())
print("SwitchSet(False) ->", oa.SwitchSet(False)); time.sleep(1.0); print("oa now:", oa.SwitchGet())
code = getattr(sc, action)(); print(f"{action} with OA off ->", code)
time.sleep(5.0)
print("SwitchSet(True) ->", oa.SwitchSet(True)); time.sleep(1.0); print("oa after:", oa.SwitchGet())
