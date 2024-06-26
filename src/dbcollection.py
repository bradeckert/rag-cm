import logging
logger = logging.getLogger(__name__)
from typing import Any
import itertools
from collections.abc import Iterator
from colbert.data import Collection
from colbert.infra.run import Run
from sqlite3 import (connect as sql_connect, Connection, Cursor)
import db
import os
import pathlib

def sql_parameter_marks(parameters: list[Any]) -> str:
    return ','.join(['?']*len(parameters))

# create (?, ?), (?, ?)
def sql_index_and_parameter_marks(parameters: list[Any]) -> str:
    return ','.join(['(?,?)']*len(parameters))

def sql_add_index_to_params(parameters: list[int]) -> list[int]:
    params_with_index = []
    indexes = list(range(0, len(parameters)))
    # create list with 2x capacity
    params_with_index = [0] * (len(parameters) * 2)
    # move all indexes into odd positions
    params_with_index[0::2] = indexes
    # move all the ids into even positions
    params_with_index[1::2] = parameters
    return params_with_index

def open_sqlite_db(database: str | os.PathLike[str], readonly:bool=False) -> Connection:
    url = f'file:{database}?mode=ro' if readonly else database
    if not readonly:
        db_path = pathlib.Path(database)
        if not db_path.exists():
            dir_path = db_path.parent
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
    # A readonly database can be shared accross multiple threads. 
    # But doing this for writable databases can lead to data-corruption. 
    con: Connection = sql_connect(url, check_same_thread=not readonly, uri=readonly)
    cursor: Cursor = con.cursor()

    # see https://charlesleifer.com/blog/going-fast-with-sqlite-and-python/
    if not readonly:
        # we only need to set this when creating the database the first time since it's a persisted property
        # but to simplify this we currently set 
        # https://sqlite.org/forum/forumpost/c4dbf6ca17
        cursor.execute('PRAGMA journal_mode=wal;')
    # 256 MB
    #con.execute('PRAGMA mmap_size=268435456;')
    cursor.execute('PRAGMA mmap_size=16777216;')
    cursor.execute('PRAGMA foreign_keys = ON;')
    #print(cursor.execute('PRAGMA compile_options').fetchall())
    (version, ) = cursor.execute('SELECT sqlite_version()').fetchone()
    (fkeys, ) = cursor.execute('PRAGMA foreign_keys').fetchone()
    logger.info(f"sqlite3 version: {version}, foreign keys: {fkeys}")
    cursor.close()
    return con

def create_tables_if_missing(con: Connection):
    cursor: Cursor = con.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.DOCUMENT}(\
                {db.ID} INTEGER PRIMARY KEY ASC\
                ,{db.FILE_NAME} TEXT\
                );")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.PASSAGE_GROUP}(\
                {db.ID} INTEGER PRIMARY KEY ASC\
                ,{db.NAME} TEXT\
                ,{db.DOC_ID} INTEGER REFERENCES {db.DOCUMENT}({db.ID}) ON DELETE CASCADE ON UPDATE CASCADE\
                );")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.PASSAGE}(\
                    {db.ID} INTEGER PRIMARY KEY ASC\
                    ,{db.CONTENT} TEXT\
                    ,{db.GROUP_ID} INTEGER REFERENCES {db.PASSAGE_GROUP}({db.ID}) ON DELETE CASCADE ON UPDATE CASCADE\
                    ,{db.PREV_ID} INTEGER REFERENCES {db.PASSAGE}({db.ID}) ON DELETE SET NULL ON UPDATE CASCADE\
                    ,{db.NEXT_ID} INTEGER REFERENCES {db.PASSAGE}({db.ID}) ON DELETE SET NULL ON UPDATE CASCADE\
                );")
    cursor.close()
    con.commit()

def add_document(cursor: Cursor, filename: str) -> int|None:
    cursor.execute(f'INSERT INTO {db.DOCUMENT} ({db.FILE_NAME}) VALUES (?)'
                   f' RETURNING {db.ID}',
                   (filename, ))
    row = cursor.fetchone()
    (doc_id, ) = row if row  is not None else (None, )
    return doc_id

def add_passage_group(cursor: Cursor, name: str, doc_id: int|None) -> int|None:
    cursor.execute(f'INSERT INTO {db.PASSAGE_GROUP} ({db.NAME}, {db.DOC_ID}) VALUES (?, ?)'
                   f' RETURNING {db.ID}',
                   (name, doc_id))
    row = cursor.fetchone()
    (group_id, ) = row if row is not None else (None, )
    return group_id

def add_passage(cursor: Cursor, passage_id: int|None, content: str, group_id: int|None, prev_passage_id: int|None) -> int|None:
    cursor.execute(f'INSERT INTO {db.PASSAGE} ({db.ID}, {db.CONTENT}, {db.GROUP_ID}, {db.PREV_ID}) VALUES (?,?,?,?)'
                   f' RETURNING {db.ID}',
                   (passage_id, content, group_id, prev_passage_id))
    row = cursor.fetchone()
    (created_passage_id, ) = row if row is not None else (None, )
    return created_passage_id

def delete_passages_by_id(cursor: Cursor, passage_ids: list[int]):
    cursor.execute(f'DELETE FROM {db.PASSAGE} AS p'
                   f' WHERE p.{db.ID} IN ({sql_parameter_marks(passage_ids)})', passage_ids)

def update_next_passage_ids(cursor: Cursor):
    # add the {db.NEXT_ID}s where they are missing
    cursor.execute(f'UPDATE {db.PASSAGE} AS p'
                   f' SET {db.NEXT_ID} = p2.{db.ID}'
                   f' FROM {db.PASSAGE} AS p2'
                   f' WHERE p2.{db.PREV_ID} = p.{db.ID}'
                   f' AND p.{db.NEXT_ID} IS NULL;')

"""
UPDATE passage AS p
SET id = seq-1
FROM (
  SELECT ROW_NUMBER() OVER(ORDER BY id) AS seq, id FROM passage 
) AS id_map
WHERE p.id = id_map.id
"""
def renumber_passage_ids(cursor: Cursor):
    # renumber all ids sequentially, closing all gaps and starting from 0
    cursor.execute(f'UPDATE {db.PASSAGE} AS p'
                   f' SET {db.ID} = seq-1'
                   f' FROM (SELECT ROW_NUMBER() OVER(ORDER BY {db.ID}) AS seq, {db.ID} FROM {db.PASSAGE}) AS id_map'
                   f' WHERE p.{db.ID} = id_map.{db.ID};')

class CursorColumnIterator(Iterator[Any]):
    def __init__(self, cursor: Cursor, column: int):
        self.cursor = cursor
        self.column = column

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        row = self.cursor.__next__()
        if row:
            value = row[self.column]
            return value
        raise StopIteration

def run_ranks() -> int:
    nranks: int =  Run().nranks or 1
    return nranks

class DbCollection(Collection):
    # TODO: We need to store the connections in a class level map. 
    # We can't store them inside the instances, since this causes an issue when ColBert is attempting to make a pickle serialization
    # This should ideally be handled differently by the ColBERT library
    connection_cache: dict[str, Cursor] = {}
    def __init__(self, db_path: str, cursor: Cursor, len: int | None = None) -> None:
        #super().__init__()
        # self.path = path
        # self.data = data or self._load_file(path)
        self.__db_path = db_path
        self.__class__.connection_cache[self.__db_path] = cursor
        #self.__cursor: Cursor = cursor
        self.__len: int | None = len

    def get_or_make_db(self) -> Cursor:
        cursor = self.__class__.connection_cache[self.__db_path]
        return cursor


    def __iter__(self) -> CursorColumnIterator:
        cursor: Cursor = self.get_or_make_db().execute(f'SELECT {db.CONTENT} FROM {db.PASSAGE}')
        return CursorColumnIterator(cursor, 0)

    def __getitem__(self, item_number: int) -> str | None:
        row = self.get_or_make_db().execute(f'SELECT * FROM {db.PASSAGE} WHERE {db.ID} = ?', (item_number,)).fetchone()
        (text, ) = row if row  is not None else (None, )
        return text
    
    def read_len(self) -> int:
        row = self.get_or_make_db().execute(f'SELECT count(*) FROM {db.PASSAGE}').fetchone()
        # also update the cached value
        (self.__len, ) = row if row is not None else (0, )
        return self.__len

    def __len__(self) -> int:
        if self.__len is None:
            return self.read_len()
        else:
            return self.__len
    
    def clear_cached_len(self):
        self.__len = None

    # def _load_file(self, path):
    #     self.path = path
    #     return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    # def _load_tsv(self, path):
    #     return load_collection(path)

    # def _load_jsonl(self, path):
    #     raise NotImplementedError()

    def provenance(self) -> str | None:
        return None
        # return self.path
    
    def toDict(self) -> Any:
        raise NotImplementedError()
        # return {'provenance': self.provenance()}

    def save(self, new_path: str) -> Any:
        raise NotImplementedError()
        # assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        # assert not os.path.exists(new_path), new_path

        # with Run().open(new_path, 'w') as f:
        #     # TODO: expects content to always be a string here; no separate title!
        #     for pid, content in enumerate(self.data):
        #         content = f'{pid}\t{content}\n'
        #         f.write(content)
            
        #     return f.name

    def enumerate(self, rank:int|None):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank:int|None, chunksize:int|None = None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(run_ranks()))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return

    def get_passages_by_id(self, passage_ids: list[int]) -> Iterator[tuple[int,str]]:
        return self.__class__.read_passages_by_id(self.get_or_make_db(), passage_ids)
    
    @classmethod
    def read_passages_by_id(cls, cursor: Cursor, passage_ids: list[int]) -> Iterator[tuple[int,str]]:
        # create (?, ?), (?, ?)
        markers_for_index_and_param = sql_index_and_parameter_marks(passage_ids)
        params_with_index = sql_add_index_to_params(passage_ids)

        query: str = (f'WITH cte(pos, id) AS (VALUES {markers_for_index_and_param})'
                      f' SELECT p.{db.ID}, {db.CONTENT}'
                      f' FROM cte c'
                      f' LEFT JOIN {db.PASSAGE} p ON p.{db.ID} = c.id'
                      ' ORDER BY c.pos')
       
        res: Cursor = cursor.execute(query, params_with_index)
        return res

    
    def get_chunksize(self) -> int:
        return min(25_000, 1 + len(self) // run_ranks())  # 25k is great, 10k allows things to reside on GPU??    
    
    @classmethod
    def cast(cls, obj: Any) -> Collection:
        if type(obj) is str:
            return Collection(path=obj)

        if type(obj) is list:
            return Collection(data=obj)

        if type(obj) is Collection:
            return obj

        if isinstance(obj, cls):
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

from colbert_patches import patch_colbert_collection_cast_be_carefull
Collection.cast = DbCollection.cast
patch_colbert_collection_cast_be_carefull()
