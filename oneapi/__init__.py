from oneapi.one_api import OneAPITool, batch_chat
from oneapi.chat import ChatAgent
from oneapi.clients import clients_register, AbstractClient, AbstractMethod
def register_client(api_type, client_cls):
    clients_register[api_type] = client_cls