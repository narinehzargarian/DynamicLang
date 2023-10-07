from intbase import InterpreterBase

class FuncInfo:
  def __init__(self, params, start_ip):
    self.params = params  
    self.start_ip = start_ip    # line number, zero-based

class FunctionManager:
  def __init__(self, tokenized_program):
    self.func_cache = {}
    self.return_types = []  # of each line in the program
    self._cache_function_parameters_and_return_type(tokenized_program)

  def get_function_info(self, func_name):
    if func_name not in self.func_cache:
      return None
    return self.func_cache[func_name]

  # returns true if the function name is a known function in the program
  def is_function(self, func_name):
    return func_name in self.func_cache

  def create_lambda_name(self, line_num):
    return InterpreterBase.LAMBDA_DEF + ':' + str(line_num)

  # returns the return type for the function in question
  def get_return_type_for_enclosing_function(self, line_num):
    return self.return_types[line_num]

  def _to_tuple(self, formal):
    var_type = formal.split(':')
    return (var_type[0], var_type[1])

  def _cache_function_parameters_and_return_type(self, tokenized_program):
    cur_return_type = None
    reset_after_this_line = False
    set_to_prev_type = False
    prev_return_type = []
    return_type_stack = [None]
    tokenized_program.append(['func', 'dummy','void'])
    tokenized_program.append(['endfunc'])
    
    for line_num, line in enumerate(tokenized_program):
      if line and line[0] == InterpreterBase.FUNC_DEF:
        func_name = line[1]
        params = [self._to_tuple(formal) for formal in line[2:-1]]
        func_info = FuncInfo(params, line_num + 1)  # function starts executing on line after funcdef
        self.func_cache[func_name] = func_info
        return_type_stack.append(line[-1])
      
      if line and line[0] == InterpreterBase.LAMBDA_DEF:
        prev_return_type.append(return_type_stack[-1])
        func_name = self.create_lambda_name(line_num)
        params = [self._to_tuple(formal) for formal in line[1:-1]]
        func_info = FuncInfo(params, line_num+1)
        self.func_cache[func_name] = func_info
        return_type_stack.append(line[-1])
      
      if line and line[0] == InterpreterBase.ENDLAMBDA_DEF:
        set_to_prev_type = True


      if line and line[0] == InterpreterBase.ENDFUNC_DEF:
        reset_after_this_line = True

      self.return_types.append(return_type_stack[-1])  

      if reset_after_this_line:  
        return_type_stack.pop()
        reset_after_this_line = False

      if set_to_prev_type:
        prev_type = prev_return_type.pop()
        return_type_stack.append(prev_type)
        set_to_prev_type = False

