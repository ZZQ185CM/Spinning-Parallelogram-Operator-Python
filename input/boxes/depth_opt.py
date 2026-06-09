"""
Depth optimization parameters
"""

# Depth range parameters
# 视差范围: 负值=远景, 正值=近景
disp_min = -2.2
disp_max = 1.4

# Options dictionary
# 与原始MATLAB版本保持一致，直接把标准视差范围映射到SPO的Dmin/Dmax
# 乘法的数字与输入光场图像的视角数有关，NumView=9时为(NumView-1)/2=4
opts = {
    'Dmin': disp_min * 4,
    'Dmax': disp_max * 4,
    'NumView': 9,
}
