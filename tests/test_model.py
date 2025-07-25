import os
import sys
from ptflops import get_model_complexity_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models import APSDCP, APSDCP_Refine


model = APSDCP().cuda()

macs, params = get_model_complexity_info(
    model,
    (3, 256, 256),
    as_strings=True,
    backend="pytorch",
    print_per_layer_stat=False,
    verbose=True,
)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))

print("=============================================")

model = APSDCP_Refine().cuda()

macs, params = get_model_complexity_info(
    model,
    (3, 256, 256),
    as_strings=True,
    backend="pytorch",
    print_per_layer_stat=False,
    verbose=True,
)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))
