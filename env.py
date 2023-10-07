from enum import Enum
import copy


class SymbolResult(Enum):
  OK = 0     # symbol created, didn't exist in top scope
  ERROR = 1  # symbol already exists in top scope

class EnvironmentManager:
  def __init__(self):
    self.environment = [[{}]]

  def get(self, symbol):
    nested_envs = self.environment[-1]
    for env in reversed(nested_envs):
      if symbol in env:
        return env[symbol]

    return None
  
  def create_new_symbol(self, symbol, create_in_top_block = False):
    block_index = 0 if create_in_top_block else -1
    if symbol not in self.environment[-1][block_index]:
      self.environment[-1][block_index][symbol] = None
      return SymbolResult.OK

    return SymbolResult.ERROR

  
  def set(self, symbol, value):
    nested_envs = self.environment[-1]

    for env in reversed(nested_envs):
      if symbol in env:
        env[symbol] = value
        return SymbolResult.OK

    return SymbolResult.ERROR

  def import_mappings(self, dict):
    cur_env = self.environment[-1][-1]
    for symbol, value in dict.items():
      cur_env[symbol] = value

  
  def capture_env_vars(self):
    top_env = self.environment[-1]
    captures = {}
    for nested_env in top_env:
      for symbol in nested_env:
        captures[symbol] = nested_env[symbol] 
    return captures

  def add_captures_to_env(self, captures):
    self.environment[-1][-1].update(captures)

  def block_nest(self):
    self.environment[-1].append({})   # [..., [{}]] -> [..., [{}, {}]]

  def block_unnest(self):
    self.environment[-1].pop()

  def push(self):
    self.environment.append([{}])       # [[...],[...]] -> [[...],[...],[]]

  def pop(self):
    self.environment.pop()
