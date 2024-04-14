from typing import List, Dict, Union
import json
from glob import glob
import os


def get_paths(path: str) -> List[str]:
    """
    Args:
        path (str): 可以是文件路径, 或者是文件夹路径

    Returns:
        List[str]: _description_
    """
    if os.path.isfile(path):  # 如果是文件, 直接返回文件路径
        return [path]
    else:  # 输入的是目录
        json_data = glob(f"{path}/**/**.json", recursive=True)
        jsonl_data = glob(f"{path}/**/**.jsonl", recursive=True)

        return json_data + jsonl_data


def loads(path: str) -> List[Dict]:
    """
    Args:
        path (str): _description_

    Returns:
        List[Dict]: _description_
    """

    datas: List[Dict] = []
    with open(path, mode="r", encoding="UTF-8") as fr:
        for line in fr:
            datas.append(json.loads(line))

    return datas


def load(path: str) -> List[Dict]:
    """
    Args:
        path (str): _description_

    Returns:
        List[Dict]: _description_
    """
    with open(path, mode="r", encoding="UTF-8") as fr:
        data = json.load(fr)

    return data


def read_datas(paths: Union[str, List[str]] = None) -> List[List[Dict]]:
    """
    Args:
        paths (Union[str, List[str]], optional): _description_. Defaults to None.

    Returns:
        List[Dict]: _description_
    """
    if not paths:
        return []

    if isinstance(paths, str):
        paths = [paths]

    datas: List[List[Dict]] = []
    for path in paths:
        fold, suffix = os.path.splitext(path)
        if ".json" == suffix:
            datas.append(load(path))
        elif ".jsonl" == suffix:
            datas.append(loads(path))

    return datas


def dump(path: str, datas: List[Dict]):
    """
    Args:
        path (str): _description_
        datas (List[Dict]): _description_
    """
    prex = path.rsplit(os.sep, 1)[0]
    os.makedirs(prex, exist_ok=True)
    with open(path, "w", encoding="UTF-8") as fw:
        json.dump(datas, fw, ensure_ascii=False, indent=4)


def dumps(path: str, datas: List[Dict]):
    """
    Args:
        path (str): _description_
        datas (List[Dict]): _description_
    """
    prex = path.rsplit(os.sep, 1)[0]
    os.makedirs(prex, exist_ok=True)
    with open(path, "w", encoding="UTF-8") as fw:
        for obj in datas:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
