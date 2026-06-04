"""
Depth optimization parameters
"""

# Depth range parameters
# 视差范围: 负值=远景, 正值=近景
disp_min = -2.2
disp_max = 1.4

# Options dictionary
# SPO算法的shift方向与标准视差定义相反，需要取反并交换min/max
# 用户设置的disp_min/disp_max是标准视差（正=近，负=远）
# 算法内部的Dmin/Dmax需要反向
# 乘法的数字与输入光场图像的分辨率有关，（NumView-1）/2可得
opts = {
    'Dmin': -disp_max * 4,  # 注意取反并交换
    'Dmax': -disp_min * 4,  # 注意取反并交换
    'NumView': 9,
}
