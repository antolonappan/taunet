import os

__path__ = os.path.dirname(os.path.realpath(__file__))

fg_path = os.path.join(__path__, '..','FG')
os.makedirs(fg_path, exist_ok=True)