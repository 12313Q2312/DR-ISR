# Measured neural comparison

Controlled synthetic experiment; not a reproduction of the paper's 16 real-world datasets.

TN-VSR and TN-DR use the paper's polynomial-neuron parameterization with a reimplemented, bounded homogeneous GP search. KAN is the attributed efficient-kan B-spline implementation, not the full pykan pipeline. Shared parameter CEILING and optimizer updates do not imply identical capacity or compute.

NMSE = MSE / variance of training labels. Data scaling, structure search and model training use training data only; validation selects checkpoints, test is used only for final evaluation. Labels remain noisy. No extrapolation experiment here.

| Dataset/group | Method | Seeds | Test NMSE median [Q1,Q3] | Parameters median | Search seconds median | Total seconds median |
|---|---|---:|---:|---:|---:|---:|
| epsilon_low/no1 | tn_dr | 5 | 0.120488 [0.03609,0.1307] | 253 | 0.953 | 2.226 |
| epsilon_low/no1 | tn_vsr | 5 | 0.266626 [0.1259,0.3205] | 253 | 0.871 | 2.009 |
| epsilon_low/no1 | mlp | 5 | 0.658957 [0.5993,0.7003] | 253 | 0.000 | 0.720 |
| epsilon_low/no1 | kan | 5 | 0.0402004 [0.04008,0.0412] | 220 | 0.000 | 3.051 |
| epsilon_low/no2 | tn_dr | 5 | 0.0604874 [0.02846,0.09035] | 253 | 0.966 | 1.918 |
| epsilon_low/no2 | tn_vsr | 5 | 0.0610733 [0.0326,0.07228] | 253 | 1.156 | 1.953 |
| epsilon_low/no2 | mlp | 5 | 0.847835 [0.8159,0.85] | 253 | 0.000 | 0.757 |
| epsilon_low/no2 | kan | 5 | 0.028605 [0.02473,0.0308] | 220 | 0.000 | 3.047 |
| epsilon_low/no3 | tn_dr | 5 | 0.10099 [0.06633,0.1017] | 254 | 1.461 | 2.602 |
| epsilon_low/no3 | tn_vsr | 5 | 0.25199 [0.07589,0.2675] | 253 | 1.284 | 2.282 |
| epsilon_low/no3 | mlp | 5 | 0.507465 [0.4619,0.5155] | 253 | 0.000 | 0.724 |
| epsilon_low/no3 | kan | 5 | 0.0361351 [0.03533,0.03762] | 220 | 0.000 | 3.185 |
| epsilon_low/no4 | tn_dr | 5 | 0.102488 [0.04328,0.1104] | 253 | 1.102 | 2.047 |
| epsilon_low/no4 | tn_vsr | 5 | 0.102422 [0.04588,0.1025] | 239 | 0.927 | 2.079 |
| epsilon_low/no4 | mlp | 5 | 0.76665 [0.7575,0.861] | 253 | 0.000 | 0.693 |
| epsilon_low/no4 | kan | 5 | 0.021471 [0.02142,0.02625] | 220 | 0.000 | 2.899 |
| epsilon_low/no5 | tn_dr | 5 | 0.0758524 [0.0504,0.09289] | 253 | 1.281 | 2.067 |
| epsilon_low/no5 | tn_vsr | 5 | 0.0758524 [0.03058,0.1036] | 239 | 1.192 | 2.457 |
| epsilon_low/no5 | mlp | 5 | 0.758233 [0.7267,0.8328] | 253 | 0.000 | 0.640 |
| epsilon_low/no5 | kan | 5 | 0.0297793 [0.02849,0.03023] | 220 | 0.000 | 2.998 |
| epsilon_low/no6 | tn_dr | 5 | 0.0768604 [0.03249,0.086] | 254 | 1.403 | 2.457 |
| epsilon_low/no6 | tn_vsr | 5 | 0.0768604 [0.03423,0.086] | 253 | 0.815 | 1.888 |
| epsilon_low/no6 | mlp | 5 | 0.686259 [0.6622,0.8365] | 253 | 0.000 | 0.740 |
| epsilon_low/no6 | kan | 5 | 0.032118 [0.02747,0.03212] | 220 | 0.000 | 3.014 |
| epsilon_low/no7 | tn_dr | 5 | 0.0745589 [0.07035,0.08056] | 253 | 1.181 | 2.030 |
| epsilon_low/no7 | tn_vsr | 5 | 0.0703451 [0.06629,0.07456] | 253 | 1.157 | 2.036 |
| epsilon_low/no7 | mlp | 5 | 0.880646 [0.7624,0.8892] | 253 | 0.000 | 0.699 |
| epsilon_low/no7 | kan | 5 | 0.0289763 [0.02855,0.03034] | 220 | 0.000 | 3.105 |
| epsilon_low/no8 | tn_dr | 5 | 0.0777237 [0.07399,0.08098] | 253 | 1.602 | 2.545 |
| epsilon_low/no8 | tn_vsr | 5 | 0.0744012 [0.03172,0.07772] | 253 | 0.871 | 1.987 |
| epsilon_low/no8 | mlp | 5 | 0.755162 [0.6935,0.8133] | 253 | 0.000 | 0.742 |
| epsilon_low/no8 | kan | 5 | 0.029303 [0.02443,0.03007] | 220 | 0.000 | 3.074 |
| epsilon_low/no9 | tn_dr | 5 | 0.00443009 [0.002904,0.004968] | 253 | 1.692 | 2.881 |
| epsilon_low/no9 | tn_vsr | 5 | 0.00329777 [0.003076,0.003903] | 254 | 1.212 | 2.057 |
| epsilon_low/no9 | mlp | 5 | 0.16376 [0.1633,0.1646] | 253 | 0.000 | 0.695 |
| epsilon_low/no9 | kan | 5 | 0.00367139 [0.003373,0.003764] | 220 | 0.000 | 3.095 |
| epsilon_low/no10 | tn_dr | 5 | 0.00408087 [0.003891,0.06145] | 254 | 1.037 | 2.349 |
| epsilon_low/no10 | tn_vsr | 5 | 0.00601108 [0.004939,0.00643] | 254 | 1.586 | 2.664 |
| epsilon_low/no10 | mlp | 5 | 0.34921 [0.3461,0.3498] | 253 | 0.000 | 0.733 |
| epsilon_low/no10 | kan | 5 | 0.0042341 [0.004088,0.006143] | 220 | 0.000 | 3.212 |
| epsilon_high/no1 | tn_dr | 5 | 0.740338 [0.5806,0.7942] | 254 | 1.188 | 2.287 |
| epsilon_high/no1 | tn_vsr | 5 | 0.584938 [0.5815,0.7403] | 239 | 1.145 | 2.534 |
| epsilon_high/no1 | mlp | 5 | 1.14033 [1.112,1.15] | 253 | 0.000 | 0.724 |
| epsilon_high/no1 | kan | 5 | 0.618811 [0.6102,0.6905] | 220 | 0.000 | 3.080 |
| epsilon_high/no2 | tn_dr | 5 | 0.4721 [0.4715,0.4789] | 254 | 1.246 | 2.373 |
| epsilon_high/no2 | tn_vsr | 5 | 0.481693 [0.4106,0.5362] | 253 | 1.231 | 2.220 |
| epsilon_high/no2 | mlp | 5 | 1.04649 [1.045,1.08] | 253 | 0.000 | 0.722 |
| epsilon_high/no2 | kan | 5 | 0.488575 [0.4793,0.5502] | 220 | 0.000 | 3.217 |
| epsilon_high/no3 | tn_dr | 5 | 0.529586 [0.5184,0.5387] | 253 | 1.020 | 2.026 |
| epsilon_high/no3 | tn_vsr | 5 | 0.529586 [0.5184,0.5387] | 253 | 0.831 | 1.771 |
| epsilon_high/no3 | mlp | 5 | 0.885693 [0.8824,0.8951] | 253 | 0.000 | 0.711 |
| epsilon_high/no3 | kan | 5 | 0.548142 [0.5345,0.5762] | 220 | 0.000 | 3.175 |
| epsilon_high/no4 | tn_dr | 5 | 0.427747 [0.409,0.4278] | 253 | 1.081 | 2.029 |
| epsilon_high/no4 | tn_vsr | 5 | 0.427787 [0.409,0.4409] | 253 | 0.901 | 1.968 |
| epsilon_high/no4 | mlp | 5 | 0.955437 [0.9384,0.9916] | 253 | 0.000 | 0.705 |
| epsilon_high/no4 | kan | 5 | 0.442573 [0.4261,0.4559] | 220 | 0.000 | 2.978 |
| epsilon_high/no5 | tn_dr | 5 | 0.539682 [0.5001,0.5467] | 253 | 1.364 | 2.368 |
| epsilon_high/no5 | tn_vsr | 5 | 0.539682 [0.5382,0.5467] | 253 | 1.079 | 2.106 |
| epsilon_high/no5 | mlp | 5 | 1.03254 [1.005,1.052] | 253 | 0.000 | 0.764 |
| epsilon_high/no5 | kan | 5 | 0.513501 [0.5047,0.5329] | 220 | 0.000 | 3.095 |
| epsilon_high/no6 | tn_dr | 5 | 0.531622 [0.5299,0.5469] | 253 | 1.085 | 2.128 |
| epsilon_high/no6 | tn_vsr | 5 | 0.559518 [0.5469,0.5643] | 253 | 0.819 | 1.772 |
| epsilon_high/no6 | mlp | 5 | 1.22693 [1.209,1.247] | 253 | 0.000 | 0.754 |
| epsilon_high/no6 | kan | 5 | 0.62879 [0.6145,0.6478] | 220 | 0.000 | 2.913 |
| epsilon_high/no7 | tn_dr | 5 | 0.470785 [0.4598,0.4739] | 253 | 1.013 | 1.870 |
| epsilon_high/no7 | tn_vsr | 5 | 0.470785 [0.4052,0.4739] | 253 | 0.948 | 1.867 |
| epsilon_high/no7 | mlp | 5 | 0.970991 [0.9215,1.035] | 253 | 0.000 | 0.753 |
| epsilon_high/no7 | kan | 5 | 0.488822 [0.4824,0.4975] | 220 | 0.000 | 3.002 |
| epsilon_high/no8 | tn_dr | 5 | 0.540663 [0.5278,0.5471] | 253 | 0.891 | 1.978 |
| epsilon_high/no8 | tn_vsr | 5 | 0.530927 [0.523,0.5407] | 253 | 0.736 | 1.663 |
| epsilon_high/no8 | mlp | 5 | 1.06507 [1.054,1.15] | 253 | 0.000 | 0.729 |
| epsilon_high/no8 | kan | 5 | 0.588509 [0.5663,0.5991] | 220 | 0.000 | 2.952 |
| epsilon_high/no9 | tn_dr | 5 | 0.0836848 [0.08367,0.09113] | 239 | 1.459 | 2.818 |
| epsilon_high/no9 | tn_vsr | 5 | 0.0789802 [0.07728,0.07922] | 253 | 1.360 | 2.345 |
| epsilon_high/no9 | mlp | 5 | 0.204557 [0.2037,0.2052] | 253 | 0.000 | 0.657 |
| epsilon_high/no9 | kan | 5 | 0.0877898 [0.08688,0.09215] | 220 | 0.000 | 2.828 |
| epsilon_high/no10 | tn_dr | 5 | 0.102882 [0.09962,0.1066] | 254 | 0.694 | 1.900 |
| epsilon_high/no10 | tn_vsr | 5 | 0.102882 [0.1005,0.1295] | 254 | 1.692 | 2.933 |
| epsilon_high/no10 | mlp | 5 | 0.408591 [0.4053,0.4092] | 253 | 0.000 | 0.696 |
| epsilon_high/no10 | kan | 5 | 0.115398 [0.1125,0.1232] | 220 | 0.000 | 3.037 |

## Paired test-NMSE differences

Negative favors the left model. Absolute tie tolerance 1e-8. Descriptive only.

| Dataset/group | Comparison | Median difference | Wins / ties / losses |
|---|---|---:|---|
| epsilon_low/no1 | tn_vsr - kan | 0.222291 | 1/0/4 |
| epsilon_low/no1 | tn_dr - kan | 0.0792857 | 2/0/3 |
| epsilon_low/no1 | tn_dr - tn_vsr | -0.0898105 | 4/1/0 |
| epsilon_low/no1 | tn_vsr - mlp | -0.468684 | 5/0/0 |
| epsilon_low/no2 | tn_vsr - kan | 0.0324684 | 1/0/4 |
| epsilon_low/no2 | tn_dr - kan | 0.0357524 | 1/0/4 |
| epsilon_low/no2 | tn_dr - tn_vsr | 0 | 2/1/2 |
| epsilon_low/no2 | tn_vsr - mlp | -0.786762 | 5/0/0 |
| epsilon_low/no3 | tn_vsr - kan | 0.209697 | 1/0/4 |
| epsilon_low/no3 | tn_dr - kan | 0.0586971 | 1/0/4 |
| epsilon_low/no3 | tn_dr - tn_vsr | -0.00956103 | 3/2/0 |
| epsilon_low/no3 | tn_vsr - mlp | -0.248033 | 5/0/0 |
| epsilon_low/no4 | tn_vsr - kan | 0.0809991 | 1/0/4 |
| epsilon_low/no4 | tn_dr - kan | 0.0810173 | 1/0/4 |
| epsilon_low/no4 | tn_dr - tn_vsr | 0 | 1/2/2 |
| epsilon_low/no4 | tn_vsr - mlp | -0.7116 | 5/0/0 |
| epsilon_low/no5 | tn_vsr - kan | 0.0473625 | 2/0/3 |
| epsilon_low/no5 | tn_dr - kan | 0.0473625 | 1/0/4 |
| epsilon_low/no5 | tn_dr - tn_vsr | 0 | 2/2/1 |
| epsilon_low/no5 | tn_vsr - mlp | -0.727654 | 5/0/0 |
| epsilon_low/no6 | tn_vsr - kan | 0.0493941 | 0/0/5 |
| epsilon_low/no6 | tn_dr - kan | 0.0493941 | 1/0/4 |
| epsilon_low/no6 | tn_dr - tn_vsr | 0 | 1/3/1 |
| epsilon_low/no6 | tn_vsr - mlp | -0.585328 | 5/0/0 |
| epsilon_low/no7 | tn_vsr - kan | 0.0400038 | 1/0/4 |
| epsilon_low/no7 | tn_dr - kan | 0.0492083 | 1/0/4 |
| epsilon_low/no7 | tn_dr - tn_vsr | 0 | 1/3/1 |
| epsilon_low/no7 | tn_vsr - mlp | -0.806088 | 5/0/0 |
| epsilon_low/no8 | tn_vsr - kan | 0.0494703 | 0/0/5 |
| epsilon_low/no8 | tn_dr - kan | 0.049563 | 1/0/4 |
| epsilon_low/no8 | tn_dr - tn_vsr | 0 | 2/2/1 |
| epsilon_low/no8 | tn_vsr - mlp | -0.724872 | 5/0/0 |
| epsilon_low/no9 | tn_vsr - kan | 7.24523e-06 | 2/0/3 |
| epsilon_low/no9 | tn_dr - kan | 0.000758696 | 2/0/3 |
| epsilon_low/no9 | tn_dr - tn_vsr | 0.000799467 | 2/0/3 |
| epsilon_low/no9 | tn_vsr - mlp | -0.159972 | 5/0/0 |
| epsilon_low/no10 | tn_vsr - kan | 0.000705107 | 2/0/3 |
| epsilon_low/no10 | tn_dr - kan | -0.000342615 | 3/0/2 |
| epsilon_low/no10 | tn_dr - tn_vsr | 0 | 2/2/1 |
| epsilon_low/no10 | tn_vsr - mlp | -0.339686 | 5/0/0 |
| epsilon_high/no1 | tn_vsr - kan | -0.0206743 | 3/0/2 |
| epsilon_high/no1 | tn_dr - kan | 0.0394739 | 2/0/3 |
| epsilon_high/no1 | tn_dr - tn_vsr | 0 | 1/2/2 |
| epsilon_high/no1 | tn_vsr - mlp | -0.542075 | 5/0/0 |
| epsilon_high/no2 | tn_vsr - kan | -0.0684173 | 4/0/1 |
| epsilon_high/no2 | tn_dr - kan | -0.07814 | 5/0/0 |
| epsilon_high/no2 | tn_dr - tn_vsr | -0.00959293 | 4/0/1 |
| epsilon_high/no2 | tn_vsr - mlp | -0.569777 | 5/0/0 |
| epsilon_high/no3 | tn_vsr - kan | -0.0466121 | 4/0/1 |
| epsilon_high/no3 | tn_dr - kan | -0.0466121 | 4/0/1 |
| epsilon_high/no3 | tn_dr - tn_vsr | 0 | 0/5/0 |
| epsilon_high/no3 | tn_vsr - mlp | -0.352803 | 5/0/0 |
| epsilon_high/no4 | tn_vsr - kan | -0.0149899 | 4/0/1 |
| epsilon_high/no4 | tn_dr - kan | -0.0161568 | 4/0/1 |
| epsilon_high/no4 | tn_dr - tn_vsr | 0 | 1/3/1 |
| epsilon_high/no4 | tn_vsr - mlp | -0.563777 | 5/0/0 |
| epsilon_high/no5 | tn_vsr - kan | 0.0138428 | 2/0/3 |
| epsilon_high/no5 | tn_dr - kan | -0.0105573 | 3/0/2 |
| epsilon_high/no5 | tn_dr - tn_vsr | 0 | 1/3/1 |
| epsilon_high/no5 | tn_vsr - mlp | -0.494354 | 5/0/0 |
| epsilon_high/no6 | tn_vsr - kan | -0.0676372 | 5/0/0 |
| epsilon_high/no6 | tn_dr - kan | -0.0989281 | 5/0/0 |
| epsilon_high/no6 | tn_dr - tn_vsr | 0 | 2/3/0 |
| epsilon_high/no6 | tn_vsr - mlp | -0.649513 | 5/0/0 |
| epsilon_high/no7 | tn_vsr - kan | -0.0236347 | 5/0/0 |
| epsilon_high/no7 | tn_dr - kan | -0.0226072 | 5/0/0 |
| epsilon_high/no7 | tn_dr - tn_vsr | 0 | 0/3/2 |
| epsilon_high/no7 | tn_vsr - mlp | -0.517487 | 5/0/0 |
| epsilon_high/no8 | tn_vsr - kan | -0.062549 | 5/0/0 |
| epsilon_high/no8 | tn_dr - kan | -0.0606845 | 5/0/0 |
| epsilon_high/no8 | tn_dr - tn_vsr | 0 | 1/2/2 |
| epsilon_high/no8 | tn_vsr - mlp | -0.534138 | 5/0/0 |
| epsilon_high/no9 | tn_vsr - kan | -0.00878037 | 5/0/0 |
| epsilon_high/no9 | tn_dr - kan | -0.008481 | 3/0/2 |
| epsilon_high/no9 | tn_dr - tn_vsr | 0.00639641 | 1/1/3 |
| epsilon_high/no9 | tn_vsr - mlp | -0.125577 | 5/0/0 |
| epsilon_high/no10 | tn_vsr - kan | -0.0125164 | 3/0/2 |
| epsilon_high/no10 | tn_dr - kan | -0.0125164 | 4/0/1 |
| epsilon_high/no10 | tn_dr - tn_vsr | 0 | 2/2/1 |
| epsilon_high/no10 | tn_vsr - mlp | -0.302399 | 5/0/0 |

All seeds and failures are retained. A shared fixed optimizer schedule is a protocol, not proof that each model was optimally tuned. Search costs are extra costs paid only by TN models, included in total time; KAN is not charged fictitious search evaluations. A single data split per synthetic task limits generalization of these findings.
