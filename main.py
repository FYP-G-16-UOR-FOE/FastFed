from typing import Optional

from fastapi import FastAPI

from client.ClientApi import app as ClientApi
from server.ServerApi import app as ServerApi

app = FastAPI()

app.mount("/server", ServerApi)
app.mount("/client", ClientApi)
