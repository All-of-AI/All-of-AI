import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

cores = cpu_count()  # cpu 数量
partitions = cores  # 分块个数


def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    """
    data_split = np.array_split(df, partitions)  # 数据切分
    pool = Pool(cores)  # 线程池
    data = pd.concat(pool.map(func, data_split))  # 数据分发 合并
    pool.close()  # 关闭线程池
    pool.join()  # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    return data
