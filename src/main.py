import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
#import asyncio
# import aiosqlite
import sqlite3
from fastapi import FastAPI, HTTPException, Form, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import markdown
import base64

from pydantic import BaseModel, EmailStr
from typing import Optional, List
import time
from typing import Any, Union
# import db

# from ragatouillefork import RAGPretrainedModel
from ColBertManager import ColBertManager
from constants import (
    DB_FILE_PATH,
    INDEX_ROOT_PATH,
    INDEX_NAME,
    # INDEX_PATH_RAGA
)
from dbcollection import (open_sqlite_db, sql_index_and_parameter_marks, sql_add_index_to_params, DbCollection)



# We make our main connection readonly, so it can be reused accross multiple threads
con_readonly: sqlite3.Connection = open_sqlite_db(DB_FILE_PATH, readonly=True)
cursor_readonly: sqlite3.Cursor = con_readonly.cursor()

db_collection = DbCollection(db_path=DB_FILE_PATH, cursor=cursor_readonly)
colbert_manager = ColBertManager(db_collection, INDEX_ROOT_PATH, INDEX_NAME)

# RAG: RAGPretrainedModel = RAGPretrainedModel.from_index(INDEX_PATH_RAGA)
# results = RAG.search(query="Test", k=3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('starting...')
    # print("starting")
    # Load resources
    # add what's needed

    yield
    logger.info('shutting down...')
    # Clean up and release the resources
    #colbert_manager.__del__()
    cursor_readonly.close()
    con_readonly.close()
    # if asyncdb_connections:
    #     await asyncdb_connections[0].close()
    return

app = FastAPI(lifespan=lifespan)

import os

# Get the absolute path to the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount the directory containing index.html as a static directory
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# asyncdb_connections: list[aiosqlite.Connection] = []

# Dependency
# Provide a re-usable way to inject a database connection into the individual
# endpoints.
# async def get_db():
#     """Return a database connection for use as a dependency.
#     This connection has the Row row factory automatically attached."""

#     db = await aiosqlite.connect(DB_FILE_PATH)
#     # Provide a smarter version of the results. This keeps from having to unpack
#     # tuples manually.
#     db.row_factory = aiosqlite.Row

#     try:
#         yield db
#     finally:
#         await db.close()



@app.get("/")
async def main():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None) -> dict[str, int | Any]:
    return {"item_id": item_id, "q": q}

@app.get("/search")
async def search(q: str, count: int=3) -> Any:
    start_search = time.time()
    results = colbert_manager.search(query=q, k=count)
    # results = RAG.search(query=q, k=count)
    elapsed_search = (time.time() - start_search)
    print(elapsed_search)
    return {"data": results}

@app.get("/querytest")
async def querytest() -> Any | list[Any]:
    start_search = time.time()
    results = list(db_collection.get_passages_by_id([7, 8, 10, 5, 9, 2, 6, 31, 53, 47]))
    elapsed_search = (time.time() - start_search)
    print(elapsed_search)
    return results

@app.get("/statictest")
async def statictest() -> Any | list[Any]:
    start_search = time.time()
    elapsed_search = (time.time() - start_search)
    print(elapsed_search)
    return "Hi this is a static result"

@app.get("/openaitest")
async def openaitest() -> Any | list[Any]:
    import os
    from openai import OpenAI
    from dotenv import load_dotenv , find_dotenv
    _ = load_dotenv(find_dotenv())

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPEN_AI_API_KEY"),
    )

    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )
    elapsed_time = (time.time() - start_time)
    print(elapsed_time)
    return chat_completion.choices[0].message.content

# async def db_test(conn: aiosqlite.Connection, passage_ids: list[int]) -> list[Any]:
#     res: list[Any] = []

#     markers_for_index_and_param = sql_index_and_parameter_marks(passage_ids)
#     params_with_index = sql_add_index_to_params(passage_ids)

#     query: str = (f'WITH cte(pos, id) AS (VALUES {markers_for_index_and_param})'
#                     f' SELECT p.{db.ID}, {db.CONTENT}'
#                     f' FROM cte c'
#                     f' LEFT JOIN {db.PASSAGE} p ON p.{db.ID} = c.id'
#                     ' ORDER BY c.pos')
       
#     #await asyncio.sleep(5)
#     async with conn.execute(query, params_with_index) as cursor:
#         res = [{'id': id, 'content': content} async for (id, content) in cursor]
#         # async for row in cursor:
#         #     res.append({'id': row[db.ID], 'content': row[db.CONTENT]})
#     return res

# @app.get("/queryaiotest")
# async def queryaio(q: str, count: int | None = None, db_connection: aiosqlite.Connection = Depends(get_db)) -> list[Any]:
#     #db_connection.row_factory = aiosqlite.Row
#     res = await db_test(db_connection, [7, 8, 10, 5, 9, 2, 6, 31, 53, 47])
#     return res

# @app.get("/queryaiotest2")
# async def queryaio2(q: str, count: int | None = None) -> list[Any]:
#     #db_connection.row_factory = aiosqlite.Row
#     start_search = time.time()
#     if not asyncdb_connections:
#         asyncdb_connections.append(await aiosqlite.connect(DB_FILE_PATH))
#     db_connection = asyncdb_connections[0]
#     res = await db_test(db_connection, [7, 8, 10, 5, 9, 2, 6, 31, 53, 47])
#     elapsed_search = (time.time() - start_search)
#     print(elapsed_search)
#     return res

# @app.get("/query")
# async def query(q: str, count: int | None = None) -> list[Any]:
#     db_con = await db
#     db_con.row_factory = aiosqlite.Row
#     res = await db_test(db_con)
#     return res

# Route to add a item
@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int) -> dict[str, str]:
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")
    # if item already exists, we'll just add the quantity.
    # get all item names
#     items_ids = {item.item_name: item.item_id if item.item_id is not None else 0 for item in grocery_list.values()}
#     if item_name in items_ids.keys():
#         # get index of item_name in item_ids, which is the item_id
#         item_id = items_ids[item_name]
#         grocery_list[item_id].quantity += quantity
# # otherwise, create a new item
#     else:
#         # generate an ID for the item based on the highest ID in the grocery_list
#         item_id = max(grocery_list.keys()) + 1 if grocery_list else 0
#         grocery_list[item_id] = ItemPayload(
#             item_id=item_id, item_name=item_name, quantity=quantity
#         )

#     return {"item": grocery_list[item_id]}
    return {"message": "OK"}

class MaintenanceRequest(BaseModel):
    first_name: str
    last_name: str
    contact_number: str
    email: EmailStr
    street_address: str
    street_address_line_2: Optional[str] = None
    city: str
    state_province: str
    postal_zip_code: str
    problem_description: str
    preferred_date_of_service: str
    
    def create_openai_query(self, files:list[Any]) -> str:
        query = (
            "A user has reported an issue with their home that needs repair. "
            "You are a digital property manager called HomeHelper working for the user. Your job is to get them a quote that they can sign off on. "
            "Take all the info given, and create the package you would send out to the home service provider that would fix the issue. "
            "You will include what type of service provider is needed, for example 'HVAC repair' or 'painter', "
            "and then detail the work that needs to be done, the approximate dimensions of the issue, "
            "the estimated hours of work it will take, and the estimated cost. "
            "Format this into a nice looking quote package that you could send to that home service provider if approved.\n\n"
            "User information: \n"
            f"Name: {self.first_name} {self.last_name}\n"
            f"Contact phone number: {self.contact_number}\n"
            f"Email address: {self.email}\n"
            f"Address: {self.street_address}\n{self.street_address_line_2}\n{self.city}, {self.state_province} {self.postal_zip_code}\n\n"
            f"Here is a quick description of the problem provided by the user:\n{self.problem_description}\n"
            f"The user has also stated that their preferred date of service for the provider to come to their home is: {self.preferred_date_of_service}\n"
        )

        if files:
            query += "The user has also provided visual content of the issue, please examine it for more detail and for dimensions of the problem.\n"
            #TODO multiple files?

        return query


async def encode_file_to_base64(file: UploadFile):
    contents = await file.read()  # Read the contents of the file
    return base64.b64encode(contents).decode('utf-8')  # Encode as base64 and convert bytes to string

@app.post("/submit-form/")
async def submit_form(  
    first_name: str = Form(...),
    last_name: str = Form(...),
    contact_number: str = Form(...),
    email: str = Form(...),
    street_address: str = Form(...),
    street_address_line_2: Optional[str] = Form(None),
    city: str = Form(...),
    state_province: str = Form(...),
    postal_zip_code: str = Form(...),
    problem_description: str = Form(...),
    preferred_date_of_service: str = Form(...),
    files: list[UploadFile] = []
    # files: Optional[List[UploadFile]] = File(default=None)  BROKEN in current fastapi https://github.com/tiangolo/fastapi/discussions/10280
    ):
    file_list = []
    if files:
        for f in files:
            file_list.append(f)

    maintenanceRequest = MaintenanceRequest(first_name=first_name, last_name=last_name, contact_number=contact_number, email=email, street_address=street_address,
                                            street_address_line_2=street_address_line_2, city=city, state_province=state_province, postal_zip_code=postal_zip_code,
                                            problem_description=problem_description, preferred_date_of_service=preferred_date_of_service)

    # send request to openAI
    import os
    from openai import OpenAI
    from dotenv import load_dotenv , find_dotenv
    _ = load_dotenv(find_dotenv())

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPEN_AI_API_KEY"),
    )

    start_time = time.time()

    msg_content: List[Any] = [
                    {
                        "type": "text",
                        "text": maintenanceRequest.create_openai_query(files=file_list),
                    },
                ]
    if file_list:
        for f in file_list:
            base64_image = await encode_file_to_base64(f)
            # print(base64_image)

            msg_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": msg_content
            }
        ],
        # model="gpt-3.5-turbo",
        model="gpt-4-turbo",
    )
    elapsed_time = (time.time() - start_time)
    print(elapsed_time)
    print(chat_completion.choices[0].message.content)
    return markdown.markdown(chat_completion.choices[0].message.content.replace("\n", "<br>")) # type: ignore


    # Here you would handle the form data, e.g., save to a database.
    # For the files, you can save them to the disk or handle as needed.
    return {"message": "Form submitted successfully!"}