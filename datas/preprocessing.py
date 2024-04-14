from typing import List, Dict, Any, Union, Optional, Tuple
from loguru import logger
from glob import glob
import os


import com


SYSTEM = """
I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me. Below is an instruction that describes a task, Write a response that appropriately completes the request.

##Instruction:
{database} contains tables such as {tables}.
{table_info}
""".strip()


def concat_fields(sqls: List[Dict[str, Any]]) -> str:
    # 表的字段和主外键
    table_info: List[str] = []
    key_info: List[str] = []
    for sql in sqls:
        TABLE_TMP = "Table {curr_table} has columns such as {fields}"
        # 获取表字段信息
        curr_table: str = list(set(sql.keys()) - set(["fileds"]))[0]
        fields: List[str] = []
        for field in sql["fileds"]["field"]:
            fields.append(field[0])
        TABLE_TMP = TABLE_TMP.format(curr_table=curr_table, fields=", ".join(fields))
        # 主键
        tmp = []
        if sql["fileds"]["foreign_key"]:
            # 当前表字段、外键表名、外键字段
            for f_k in sql["fileds"]["foreign_key"]:
                # 这里不是很严谨
                cur_field, f_table_name, f_field = f_k[:3]
                tmp.append(f"{cur_field} is the primary key")
            TABLE_TMP += f', where {", ".join(tmp)}'
        table_info.append(TABLE_TMP + ".")

        # 获取表的主外键
        for f_k in sql["fileds"]["foreign_key"]:
            cur_field, f_table_name, f_field = f_k[:3]
            key_info.append(
                f"The {f_field} of {f_table_name} is the foreign key of {cur_field} of {curr_table}."
            )

    return "\n".join(table_info + key_info)


def dialog(
    database: str,
    tables_name: List[str],
    query: str,
    sqls: List[Dict[str, Any]],
    answer: str,
    insert: List[str],
) -> Dict[str, Any]:
    dialog_sqls: List[Dict[str, Any]] = []
    # 选择出当前对话所需要的 sql
    for sql in sqls:
        curr_table_name = list(set(sql.keys()) - set(["fileds"]))[0]
        if curr_table_name in tables_name:
            dialog_sqls.append(sql)

    # 从表中提取出表的字段, 主外键
    table_info: str = concat_fields(dialog_sqls)
    system = SYSTEM.format(
        database=database, tables=", ".join(tables_name), table_info=table_info
    )
    return {
        "database": database,
        "system": system,
        "query": query,
        "answer": answer,
        "insert": insert,
    }


if __name__ == "__main__":
    paths = glob(
        "/root/autodl-tmp/Baichuan2-our/datas/origin/**/*.json", recursive=True
    )
    for i, path in enumerate(paths):
        logger.info(f"{i}, {path}")
        fold = path.split(os.sep)[-2]

        data = com.load(path)
        sql = dialog(
            data["database"],
            data["table_names"],
            data["input"],
            data["sql"],
            data["response"],
            data["insert"],
        )
        com.dump(
            f'/root/autodl-tmp/Baichuan2-our/datas/handled/{fold}/{sql["database"]}/{str(i)}.json',
            {
                "conversations": [
                    {
                        "database": sql["database"],
                        "system": sql["system"],
                        "query": sql["query"],
                        "answer": sql["answer"],
                        "insert": sql["insert"],
                    }
                ]
            },
        )
