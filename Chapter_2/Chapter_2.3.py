import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# 1. 定义远程下载地址和本地保存路径
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    自动化下载并解压数据的函数
    """
    # 如果本地目录不存在，则创建它
    os.makedirs(housing_path, exist_ok=True)

    # 拼接本地压缩包的完整路径
    tgz_path = os.path.join(housing_path, "housing.tgz")

    # 从 URL 下载文件并保存到本地
    print("正在下载数据...")
    urllib.request.urlretrieve(housing_url, tgz_path)

    # 打开压缩包并解压所有内容到指定目录
    print("正在解压数据...")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    print(f"数据已就绪，保存在: {housing_path}")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(housing):
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # 根据直方图划分数据段，计算比例后，使用split方法进行分层抽样，确保取样的全局代表性
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set

def matplot(strat_train_set):
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()


if __name__ == "__main__":
    # fetch_housing_data()
    housing = load_housing_data()
    print(split_train_test(housing))