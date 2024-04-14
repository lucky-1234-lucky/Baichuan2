from typing import List, Dict, Union, Any, Optional
from glob import glob
import json
import os


def get_paths(path: Optional[str] = None) -> List[str]:
    """
    Args:
        path: 文件夹或文件路径, 支持正则表达式

    Return:
        只返回*.json和*.jsonl文件的路径
    """
    if not path:
        return []

    if os.path.isfile(path):
        return [path]

    paths = glob(f"{path}/**/*.json", recursive=True) + glob(
        f"{path}/**/*.jsonl", recursive=True
    )
    return paths


def loads(path: str) -> List[Dict[str, Any]]:
    datas: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="UTF-8") as fr:
        for line in fr:
            datas.append(json.loads(line))

    return datas


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, mode="r", encoding="UTF-8") as fr:
        data = json.load(fr)

    return data


def read_datas(path: str) -> List[Dict[str, Any]]:
    """
    Args:
        path: 文件夹或文件路径, 支持正则表达式

    Return:
        只返回*.json和*.jsonl文件路径的数据
    """
    datas: List[Dict[str, Any]] = []
    for path_ in get_paths(path):
        fold, suffix = os.path.splitext(path_)
        if ".json" == suffix:
            datas.append(load(path_))
        elif ".jsonl" == suffix:
            datas.extend(loads(path_))

    return datas


def dump(path: str, datas: List[Any]):
    prex = path.rsplit(os.sep, 1)[0]
    os.makedirs(prex, exist_ok=True)
    with open(path, "w", encoding="UTF-8") as fw:
        json.dump(datas, fw, ensure_ascii=False, indent=4)


def dumps(path: str, datas: List[Any]):
    prex = path.rsplit(os.sep, 1)[0]
    os.makedirs(prex, exist_ok=True)
    with open(path, "w", encoding="UTF-8") as fw:
        for obj in datas:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
