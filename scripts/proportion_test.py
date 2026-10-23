import math

# Previous (archived) dataset stats (from earlier inspection)
n_prev = 481453
p_prev = 0.611962

# Current dataset stats (just inspected)
n_curr = 480557
p_curr = 0.591147

# Two-proportion z-test
p_pool = (p_prev * n_prev + p_curr * n_curr) / (n_prev + n_curr)
se = math.sqrt(p_pool * (1 - p_pool) * (1.0/n_prev + 1.0/n_curr))

z = (p_prev - p_curr) / se if se > 0 else float('nan')
# two-sided p-value
from math import erfc
p_value = erfc(abs(z) / math.sqrt(2))

print(f"prev: n={n_prev}, p={p_prev:.6f}")
print(f"curr: n={n_curr}, p={p_curr:.6f}")
print(f"difference = {p_prev - p_curr:.6f} (~{(p_prev-p_curr)*100:.3f} percentage points)")
print(f"z = {z:.3f}, p-value (two-sided) = {p_value:.3e}")

# Practical significance: compute 95% CI for current p
se_curr = math.sqrt(p_curr*(1-p_curr)/n_curr)
ci_low = p_curr - 1.96*se_curr
ci_high = p_curr + 1.96*se_curr
print(f"current 95% CI = [{ci_low:.4f}, {ci_high:.4f}] -> [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")